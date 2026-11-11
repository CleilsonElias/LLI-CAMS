"""Local HTTP application, JSON API and asynchronous Omega worker."""
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
import hmac
import json
import math
import threading
from urllib.parse import urlsplit
import uuid
from .llm import ask, ExtractiveBackend
from .storage import CapacityError, ConflictError


class Coordinator:
    def __init__(self, engine, backend, sleep_interval=0):
        self.engine, self.backend = engine, backend
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="llicams-omega")
        self.jobs, self.lock = {}, threading.Lock()
        self.stop = threading.Event()
        self.scheduler = None
        if sleep_interval > 0:
            def periodic():
                while not self.stop.wait(sleep_interval):
                    if self.engine.stats()["pending"]:
                        self.start_sleep()
            self.scheduler = threading.Thread(target=periodic, daemon=True, name="llicams-scheduler")
            self.scheduler.start()

    def start_sleep(self):
        with self.lock:
            for identifier, future in self.jobs.items():
                if not future.done():
                    return identifier
            identifier = str(uuid.uuid4())
            self.jobs[identifier] = self.pool.submit(self.engine.sleep)
            for old in list(self.jobs)[:-20]:
                if self.jobs[old].done():
                    del self.jobs[old]
            return identifier

    def job(self, identifier):
        with self.lock:
            future = self.jobs.get(identifier)
        if future is None:
            raise KeyError("Job not found")
        if not future.done():
            return {"id": identifier, "status": "running"}
        try:
            return {"id": identifier, "status": "completed", "result": future.result()}
        except Exception as exc:
            return {"id": identifier, "status": "failed", "error": str(exc)}

    def close(self):
        self.stop.set()
        if self.scheduler:
            self.scheduler.join()
        self.pool.shutdown(wait=True, cancel_futures=True)


def create_server(engine, backend=None, *, host="127.0.0.1", port=8765, token=None, sleep_interval=0):
    if not math.isfinite(sleep_interval) or sleep_interval < 0:
        raise ValueError("sleep_interval must be finite and non-negative")
    if host not in {"127.0.0.1", "localhost", "::1"} and not token:
        raise ValueError("Binding beyond loopback requires LLICAMS_API_TOKEN")
    coordinator = Coordinator(engine, backend or ExtractiveBackend(), sleep_interval)

    class Handler(BaseHTTPRequestHandler):
        server_version = "LLICAMS/0.1"

        def setup(self):
            super().setup()
            self.connection.settimeout(30)

        def log_message(self, *args):
            pass  # Memory text and questions are never written to HTTP access logs.

        def send(self, status, payload, content_type="application/json; charset=utf-8"):
            data = (json.dumps(payload, ensure_ascii=False, allow_nan=False).encode()
                    if content_type.startswith("application/json") else payload)
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(data)

        def authorized(self):
            if token:
                supplied = self.headers.get("Authorization", "")
                if not hmac.compare_digest(supplied, "Bearer " + token):
                    self.send(401, {"error": "Authentication required"})
                    return False
            return True

        def do_GET(self):
            path = urlsplit(self.path).path
            if path == "/":
                return self.send(200, files("llicams").joinpath("ui.html").read_bytes(), "text/html; charset=utf-8")
            if not self.authorized():
                return
            try:
                if path == "/api/health":
                    return self.send(200, {"ok": True, "backend": coordinator.backend.name})
                if path == "/api/state":
                    view = engine.store.view()
                    return self.send(200, {"stats": engine.stats(), "backend": coordinator.backend.name,
                        "records": [m.data() for m in view["records"]], "edges": view["edges"],
                        "proposals": engine.store.proposals()})
                if path == "/api/export":
                    return self.send(200, engine.export())
                if path.startswith("/api/jobs/"):
                    return self.send(200, coordinator.job(path.rsplit("/", 1)[1]))
                return self.send(404, {"error": "Route not found"})
            except KeyError as exc:
                return self.send(404, {"error": str(exc)})

        def do_POST(self):
            if not self.authorized():
                return
            origin = self.headers.get("Origin")
            if origin and urlsplit(origin).netloc != self.headers.get("Host"):
                return self.send(403, {"error": "Cross-origin writes are disabled"})
            if self.headers.get("Content-Type", "").split(";", 1)[0] != "application/json":
                return self.send(415, {"error": "Use application/json"})
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 1_048_576:
                    return self.send(413, {"error": "Body must contain 1..1048576 bytes"})
                body = json.loads(self.rfile.read(size))
                if not isinstance(body, dict):
                    raise ValueError("JSON body must be an object")
                path = urlsplit(self.path).path
                if path == "/api/observe":
                    record = engine.observe(body["text"], source=body.get("source", "interface"),
                        confidence=body.get("confidence", 1), metadata=body.get("metadata"))
                    return self.send(201, record.data())
                if path == "/api/query":
                    result = engine.retrieve(body["query"], k=body.get("k", 5))
                    return self.send(200, {**result, "hits": [h.data() for h in result["hits"]]})
                if path == "/api/chat":
                    return self.send(200, ask(engine, body["query"], coordinator.backend, body.get("k", 5)))
                if path == "/api/sleep":
                    return self.send(202, {"job_id": coordinator.start_sleep()})
                if path == "/api/promote":
                    return self.send(200, engine.promote(body["id"], body["evidence_note"]))
                if path == "/api/endogenous":
                    return self.send(200, {"proposals": engine.endogenous(body.get("count", 3))})
                return self.send(404, {"error": "Route not found"})
            except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
                return self.send(400, {"error": str(exc)})
            except (CapacityError, ConflictError) as exc:
                return self.send(409, {"error": str(exc)})
            except RuntimeError as exc:
                return self.send(503, {"error": str(exc)})

    try:
        server = ThreadingHTTPServer((host, port), Handler)
    except BaseException:
        coordinator.close()
        raise
    server.daemon_threads = True
    server.coordinator = coordinator
    return server


def serve(engine, backend=None, host="127.0.0.1", port=8765, token=None, sleep_interval=0):
    server = create_server(engine, backend, host=host, port=port, token=token, sleep_interval=sleep_interval)
    print(f"LLI-CAMS: http://{host}:{server.server_address[1]}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        server.coordinator.close()
