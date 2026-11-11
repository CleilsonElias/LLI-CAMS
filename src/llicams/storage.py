"""SQLite persistence and compare-and-swap publication of memory snapshots."""
from contextlib import contextmanager
import json
from pathlib import Path
import sqlite3
import threading
import time
import numpy as np
from .types import Memory


class ConflictError(RuntimeError):
    pass


class CapacityError(RuntimeError):
    pass


def json_text(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False,
                      separators=(",", ":"))


class Store:
    def __init__(self, path, config, encoder_fingerprint):
        self.path = str(path)
        if self.path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.config = config
        self.lock = threading.RLock()
        self.db = sqlite3.connect(self.path, timeout=30, isolation_level=None, check_same_thread=False)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA secure_delete=ON")
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS records(
                id TEXT PRIMARY KEY, seq INTEGER NOT NULL UNIQUE, state TEXT NOT NULL,
                data TEXT NOT NULL, vector BLOB NOT NULL, text_bytes INTEGER NOT NULL);
            CREATE INDEX IF NOT EXISTS records_state ON records(state,seq);
            CREATE TABLE IF NOT EXISTS edges(a TEXT,b TEXT,weight REAL,PRIMARY KEY(a,b));
            CREATE TABLE IF NOT EXISTS events(id INTEGER PRIMARY KEY AUTOINCREMENT, data TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS proposals(id INTEGER PRIMARY KEY AUTOINCREMENT,data TEXT NOT NULL);
        """)
        expected = {"schema": 1, "config": config.to_dict(), "encoder": encoder_fingerprint}
        try:
            with self.transaction():
                existing = self._get("identity")
                if existing is None:
                    self._set("identity", expected)
                    for key, value in (("version", 0), ("seq", 0), ("hr_count", 0), ("hr_total", 0.0),
                                       ("weight_artifact", None)):
                        self._set(key, value)
                elif existing != expected:
                    raise ValueError("Database configuration/encoder differs; use its original configuration or a new DB")
        except BaseException:
            self.db.close()
            raise

    @contextmanager
    def transaction(self, write=True):
        with self.lock:
            self.db.execute("BEGIN IMMEDIATE" if write else "BEGIN")
            try:
                yield
            except BaseException:
                if self.db.in_transaction:
                    self.db.execute("ROLLBACK")
                raise
            else:
                self.db.execute("COMMIT")

    def _get(self, key):
        row = self.db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def _set(self, key, value):
        self.db.execute("INSERT OR REPLACE INTO meta VALUES (?,?)", (key, json_text(value)))

    def _event(self, kind, payload):
        self.db.execute("INSERT INTO events(data) VALUES (?)",
                        (json_text({"kind": kind, "time": time.time(), **payload}),))
        self.db.execute("DELETE FROM events WHERE id NOT IN (SELECT id FROM events ORDER BY id DESC LIMIT ?)",
                        (self.config.max_events,))

    def _put(self, record):
        data = record.data()
        vector = np.asarray(record.vector, dtype="<f4")
        self.db.execute("INSERT INTO records VALUES (?,?,?,?,?,?)",
                        (record.id, record.seq, record.state, json_text(data), vector.tobytes(),
                         len((record.text or "").encode())))

    def _decode(self, row):
        data = json.loads(row[0])
        vector = np.frombuffer(row[1], dtype="<f4").copy()
        if vector.shape != (self.config.dimension,) or not np.isfinite(vector).all():
            raise ValueError("Corrupted or incompatible stored vector")
        vector.setflags(write=False)
        data["probabilities"] = tuple(data["probabilities"])
        return Memory(vector=vector, **data)

    def append(self, record):
        with self.transaction():
            if self.db.execute("SELECT 1 FROM records WHERE id=?", (record.id,)).fetchone():
                raise ValueError("Memory id already exists")
            pending = self.db.execute("SELECT COUNT(*) FROM records WHERE state='pending'").fetchone()[0]
            used = self.db.execute("SELECT COALESCE(SUM(text_bytes),0) FROM records").fetchone()[0]
            if pending >= self.config.max_pending:
                raise CapacityError("Pending buffer full; run a sleep cycle")
            if used + len((record.text or "").encode()) > self.config.max_total_text_bytes:
                raise CapacityError("Text budget exhausted; consolidate, forget memories, or choose a larger budget")
            seq = self._get("seq") + 1
            from dataclasses import replace
            record = replace(record, seq=seq)
            self._put(record)
            self._set("seq", seq)
            self._set("hr_count", self._get("hr_count") + 1)
            self._set("hr_total", self._get("hr_total") + record.hr_normalized)
            self._event("observed", {"id": record.id, "seq": seq})
        return record

    def view(self):
        with self.transaction(write=False):
            records = [self._decode(r) for r in self.db.execute("SELECT data,vector FROM records ORDER BY seq")]
            return {"records": records, "version": self._get("version"), "seq": self._get("seq"),
                    "hr_count": self._get("hr_count"), "hr_total": self._get("hr_total"),
                    "weight_artifact": self._get("weight_artifact"),
                    "edges": [tuple(e) for e in self.db.execute("SELECT a,b,weight FROM edges ORDER BY a,b")]}

    def commit(self, base_version, cutoff, retained, edges, report, proposals=(), weight_artifact=None):
        """Retain concurrent observations with seq > cutoff; publish memory and artifact atomically."""
        with self.transaction():
            if self._get("version") != base_version:
                raise ConflictError("Snapshot changed during preparation; no candidate state was published")
            self.db.execute("DELETE FROM records WHERE state!='pending' OR seq<=?", (cutoff,))
            for record in retained:
                if record.state not in {"active", "shadow"}:
                    raise ValueError("A published snapshot contains only active/shadow records")
                self._put(record)
            self.db.execute("DELETE FROM edges")
            self.db.executemany("INSERT INTO edges VALUES (?,?,?)", edges)
            for proposal in proposals:
                self.db.execute("INSERT INTO proposals(data) VALUES (?)", (json_text(proposal),))
            self.db.execute("DELETE FROM proposals WHERE id NOT IN (SELECT id FROM proposals ORDER BY id DESC LIMIT ?)",
                            (self.config.max_proposals,))
            self._set("version", base_version + 1)
            self._set("weight_artifact", weight_artifact)
            self._event("published", {"version": base_version + 1, "cutoff": cutoff, "report": report})
        return base_version + 1

    def events(self):
        with self.lock:
            return [json.loads(x[0]) for x in self.db.execute("SELECT data FROM events ORDER BY id")]

    def proposals(self):
        with self.lock:
            return [{"id": r[0], **json.loads(r[1])} for r in self.db.execute("SELECT id,data FROM proposals ORDER BY id")]

    def backup(self, destination):
        if str(destination) == self.path or (self.path != ":memory:" and Path(destination).resolve() == Path(self.path).resolve()):
            raise ValueError("Backup destination must differ from the live database")
        Path(destination).parent.mkdir(parents=True, exist_ok=True)
        with self.lock, sqlite3.connect(destination) as target:
            self.db.backup(target)

    def forget(self, memory_id):
        with self.transaction():
            if not self.db.execute("SELECT 1 FROM records WHERE id=?", (memory_id,)).fetchone():
                raise ValueError("Memory id not found")
            self.db.execute("DELETE FROM records WHERE id=?", (memory_id,))
            self.db.execute("DELETE FROM edges WHERE a=? OR b=?", (memory_id, memory_id))
            self._set("version", self._get("version") + 1)
            self._event("forgotten", {"id": memory_id, "parametric_unlearning": False})
            return {"id": memory_id, "version": self._get("version"), "parametric_unlearning": False}

    def close(self):
        with self.lock:
            self.db.close()
