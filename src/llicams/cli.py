import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import sqlite3
import sys
from .config import Config
from .engine import Engine
from .encoders import SentenceTransformerEncoder
from .llm import ExtractiveBackend, OllamaBackend, TransformersBackend, LLMREM, ask


def output(value, destination=None):
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
    if destination:
        Path(destination).parent.mkdir(parents=True, exist_ok=True)
        Path(destination).write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


def open_engine(args):
    existing = Path(args.db).is_file() and Path(args.db).stat().st_size > 0
    if args.config:
        config = Config.from_dict(json.loads(Path(args.config).read_text(encoding="utf-8")))
    elif existing:
        with sqlite3.connect(args.db) as db:
            row = db.execute("SELECT value FROM meta WHERE key='identity'").fetchone()
        if not row:
            raise ValueError("Database does not contain a LLI-CAMS identity")
        config = Config.from_dict(json.loads(row[0])["config"])
    else:
        config = Config()
    encoder = None
    if args.encoder_model:
        encoder = SentenceTransformerEncoder(args.encoder_model, revision=args.encoder_revision,
            device=args.device, local_files_only=not args.allow_download)
        if not existing and not args.config:
            config = replace(config, dimension=encoder.dimension)
    if args.backend != "extractive" and not args.model:
        raise ValueError("--model is required for a generative backend")
    if args.backend == "ollama":
        backend = OllamaBackend(args.model, args.base_url)
    elif args.backend == "transformers":
        backend = TransformersBackend(args.model, revision=args.model_revision,
                                     device=args.device, local_files_only=not args.allow_download)
    else:
        backend = ExtractiveBackend()
    replay = None
    if args.replay_model:
        from .replay import HFLoRAReplay
        replay = HFLoRAReplay(args.replay_model, args.adapter_directory, revision=args.model_revision,
            rank=args.lora_rank, steps=args.lora_steps, device=args.device,
            local_files_only=not args.allow_download)
    if args.rem_llm and args.backend == "extractive":
        raise ValueError("--rem-llm requires a generative backend")
    engine = Engine(args.db, config, encoder=encoder, replay=replay,
                    rem=LLMREM(backend) if args.rem_llm else None)
    return engine, backend


def parser():
    p = argparse.ArgumentParser(prog="llicams", description="LLI-CAMS reference application")
    p.add_argument("--db", default="llicams.sqlite")
    p.add_argument("--config", help="JSON Config; an existing DB otherwise reloads its stored configuration")
    p.add_argument("--encoder-model", help="Optional sentence-transformers model; specify again when reopening its DB")
    p.add_argument("--encoder-revision")
    p.add_argument("--backend", choices=["extractive", "ollama", "transformers"], default="extractive")
    p.add_argument("--model")
    p.add_argument("--model-revision")
    p.add_argument("--device", default="cpu")
    p.add_argument("--base-url", default="http://127.0.0.1:11434")
    p.add_argument("--allow-download", action="store_true", help="Explicitly permit model downloads")
    p.add_argument("--replay-model", help="Enable optional real LoRA replay on replay_approved text")
    p.add_argument("--adapter-directory", default="adapters")
    p.add_argument("--lora-rank", type=int, default=8)
    p.add_argument("--lora-steps", type=int, default=20)
    p.add_argument("--rem-llm", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("init", help="Create/check a persistent memory database")
    ob = sub.add_parser("observe")
    ob.add_argument("text")
    ob.add_argument("--source", default="manual")
    ob.add_argument("--metadata", default="{}")
    ing = sub.add_parser("ingest", help="Ingest a UTF-8 JSONL with text/source/metadata fields")
    ing.add_argument("file")
    ing.add_argument("--sleep-every", type=int, default=32)
    sub.add_parser("sleep")
    q = sub.add_parser("query")
    q.add_argument("text")
    q.add_argument("--k", type=int, default=5)
    ch = sub.add_parser("chat")
    ch.add_argument("text")
    sub.add_parser("stats")
    sub.add_parser("state")
    export = sub.add_parser("export")
    export.add_argument("--output", required=True)
    backup = sub.add_parser("backup", help="Consistent SQLite backup; does not include external adapter directories")
    backup.add_argument("--output", required=True)
    forget = sub.add_parser("forget")
    forget.add_argument("id")
    promote = sub.add_parser("promote")
    promote.add_argument("id")
    promote.add_argument("--evidence-note", required=True)
    endogenous = sub.add_parser("endogenous")
    endogenous.add_argument("--count", type=int, default=3)
    server = sub.add_parser("serve")
    server.add_argument("--host", default="127.0.0.1")
    server.add_argument("--port", type=int, default=8765)
    server.add_argument("--sleep-interval", type=float, default=0, help="Seconds between automatic consolidation checks; 0 disables scheduling")
    demo = sub.add_parser("demo")
    demo.add_argument("--directory", default="demo_run")
    bench = sub.add_parser("benchmark")
    bench.add_argument("--seeds", type=int, default=5)
    bench.add_argument("--observations", type=int, default=160)
    bench.add_argument("--queries", type=int, default=48)
    bench.add_argument("--output")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command == "demo":
            from .demo import run_demo
            output(run_demo(args.directory))
            return 0
        if args.command == "benchmark":
            from .benchmark import run_benchmark
            output(run_benchmark(args.seeds, args.observations, args.queries), args.output)
            return 0
        engine, backend = open_engine(args)
        with engine:
            if args.command == "init":
                output({"config": engine.config.to_dict(), "stats": engine.stats()})
            elif args.command == "observe":
                output(engine.observe(args.text, source=args.source, metadata=json.loads(args.metadata)).data())
            elif args.command == "ingest":
                if args.sleep_every < 1:
                    raise ValueError("sleep-every must be positive")
                count = 0
                with Path(args.file).open(encoding="utf-8") as source:
                    for line in source:
                        if not line.strip():
                            continue
                        row = json.loads(line)
                        engine.observe(row["text"], source=row.get("source", args.file),
                            confidence=row.get("confidence", 1), metadata=row.get("metadata"),
                            memory_id=row.get("id"))
                        count += 1
                        if count % args.sleep_every == 0:
                            engine.sleep()
                if count % args.sleep_every:
                    engine.sleep()
                output({"ingested": count, "stats": engine.stats()})
            elif args.command == "sleep":
                output(engine.sleep())
            elif args.command == "query":
                result = engine.retrieve(args.text, k=args.k)
                output({**result, "hits": [h.data() for h in result["hits"]]})
            elif args.command == "chat":
                output(ask(engine, args.text, backend))
            elif args.command == "stats":
                output(engine.stats())
            elif args.command == "state":
                output({"stats": engine.stats(), "records": [m.data() for m in engine.store.view()["records"]]})
            elif args.command == "export":
                output(engine.export(), args.output)
            elif args.command == "backup":
                engine.store.backup(args.output)
                output({"backup": args.output})
            elif args.command == "forget":
                output(engine.store.forget(args.id))
            elif args.command == "promote":
                output(engine.promote(args.id, args.evidence_note))
            elif args.command == "endogenous":
                output(engine.endogenous(args.count))
            elif args.command == "serve":
                from .service import serve
                serve(engine, backend, host=args.host, port=args.port, token=os.environ.get("LLICAMS_API_TOKEN"),
                      sleep_interval=args.sleep_interval)
        return 0
    except (ValueError, RuntimeError, OSError, sqlite3.Error) as exc:
        print(f"llicams: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
