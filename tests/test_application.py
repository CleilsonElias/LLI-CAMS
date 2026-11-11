from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import urllib.request
import urllib.error
from llicams import Config, Engine
from llicams.llm import OllamaBackend, LLMREM
from llicams.service import create_server
from llicams.benchmark import run_benchmark


class ApplicationTests(unittest.TestCase):
    def setUp(self):
        self.engine = Engine()
        self.server = create_server(self.engine, port=0)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.server.coordinator.close()
        self.thread.join(2)
        self.engine.close()

    def request(self, path, payload=None, headers=None):
        data = json.dumps(payload).encode() if payload is not None else None
        headers = {"Content-Type": "application/json", **(headers or {})}
        with urllib.request.urlopen(urllib.request.Request(self.base + path, data, headers), timeout=5) as response:
            return response.status, json.load(response)

    def test_http_observe_sleep_query_export(self):
        status, memory = self.request('/api/observe', {"text": "Código fictício de laboratório AX-991", "metadata": {"pinned": True}})
        self.assertEqual(status, 201)
        _, job = self.request('/api/sleep', {})
        for _ in range(100):
            _, state = self.request('/api/jobs/' + job['job_id'])
            if state['status'] != 'running': break
            time.sleep(.01)
        self.assertEqual(state['status'], 'completed')
        _, answer = self.request('/api/chat', {"query": "Código de laboratório AX-991"})
        self.assertIn('AX-991', answer['answer'])
        self.assertEqual(answer['backend'], 'extractive')
        _, exported = self.request('/api/export')
        self.assertEqual(exported['records'][0]['id'], memory['id'])

    def test_serves_offline_ui_without_external_dependencies(self):
        with urllib.request.urlopen(self.base + '/') as response:
            html = response.read().decode()
        self.assertIn('<html lang="pt-BR">', html)
        self.assertNotIn('<script src="http', html)
        self.assertIn('textContent', html)

    def test_rejects_cross_origin_writes(self):
        with self.assertRaises(urllib.error.HTTPError) as caught:
            self.request('/api/observe', {'text': 'x'}, {'Origin': 'http://unrelated.example'})
        self.assertEqual(caught.exception.code, 403)

    def test_rejects_wrong_content_type(self):
        with self.assertRaises(urllib.error.HTTPError) as caught:
            self.request('/api/observe', {'text': 'x'}, {'Content-Type': 'text/plain'})
        self.assertEqual(caught.exception.code, 415)

    def test_rejects_invalid_json_fields(self):
        with self.assertRaises(urllib.error.HTTPError) as caught:
            self.request('/api/observe', {'metadata': []})
        self.assertEqual(caught.exception.code, 400)


class BackendTests(unittest.TestCase):
    def test_optional_bearer_token_is_enforced(self):
        with Engine() as engine:
            server = create_server(engine, port=0, token='test-only-token')
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base = f'http://127.0.0.1:{server.server_address[1]}'
            try:
                with self.assertRaises(urllib.error.HTTPError) as caught:
                    urllib.request.urlopen(base + '/api/health')
                self.assertEqual(caught.exception.code, 401)
                request = urllib.request.Request(base + '/api/health', headers={'Authorization': 'Bearer test-only-token'})
                with urllib.request.urlopen(request) as response:
                    self.assertTrue(json.load(response)['ok'])
            finally:
                server.shutdown(); server.server_close(); server.coordinator.close(); thread.join(2)

    def test_automatic_sleep_consolidates_pending(self):
        with Engine() as engine:
            server = create_server(engine, port=0, sleep_interval=.01)
            try:
                engine.observe('Scheduled synthetic fact ZX-922', metadata={'pinned': True})
                deadline = time.monotonic() + 3
                while engine.stats()['pending'] and time.monotonic() < deadline:
                    time.sleep(.01)
                self.assertEqual(engine.stats()['pending'], 0)
                self.assertEqual(engine.stats()['active'], 1)
            finally:
                server.server_close(); server.coordinator.close()

    def test_remote_bind_requires_explicit_token(self):
        with Engine() as engine:
            with self.assertRaises(ValueError):
                create_server(engine, host='0.0.0.0', port=0)

    def test_ollama_protocol_against_local_http_fixture(self):
        captured = []
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args): pass
            def do_POST(self):
                captured.append((self.path, json.loads(self.rfile.read(int(self.headers['Content-Length'])))))
                data = json.dumps({'message': {'content': 'Resposta de teste'}}).encode()
                self.send_response(200)
                self.send_header('Content-Length', str(len(data)))
                self.end_headers()
                self.wfile.write(data)
        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            backend = OllamaBackend('fixture-model', f'http://127.0.0.1:{server.server_address[1]}')
            self.assertEqual(backend.complete([{'role': 'user', 'content': 'teste'}]), 'Resposta de teste')
            self.assertEqual(captured[0][0], '/api/chat')
            self.assertFalse(captured[0][1]['stream'])
        finally:
            server.shutdown(); server.server_close(); thread.join(2)

    def test_rem_invalid_critic_response_stays_unverified(self):
        class Backend:
            def complete(self, messages, **kwargs): return 'not JSON'
        class Record:
            text = 'input'
        proposal = LLMREM(Backend()).propose(Record(), Record())
        self.assertEqual(proposal['status'], 'unverified')
        self.assertEqual(proposal['coherence'], 0)

    def test_backup_and_forget_preserve_integrity(self):
        with tempfile.TemporaryDirectory() as folder:
            original, backup = Path(folder)/'state.sqlite', Path(folder)/'backup.sqlite'
            with Engine(original) as engine:
                record = engine.observe('fictício AX-992', metadata={'pinned': True})
                engine.sleep()
                engine.store.backup(backup)
                engine.store.forget(record.id)
                self.assertEqual(engine.stats()['active'], 0)
            with Engine(backup) as copied:
                self.assertEqual(copied.stats()['active'], 1)

    def test_cli_round_trip(self):
        with tempfile.TemporaryDirectory() as folder:
            db = str(Path(folder) / 'cli.sqlite')
            def run(*arguments):
                launcher = Path(__file__).resolve().parents[1] / 'run_llicams.py'
                p = subprocess.run([sys.executable, str(launcher), '--db', db, *arguments],
                                   capture_output=True, text=True, timeout=20)
                self.assertEqual(p.returncode, 0, p.stderr)
                return json.loads(p.stdout)
            run('init')
            run('observe', 'Código fictício TZ-628', '--metadata', '{"pinned":true}')
            run('sleep')
            response = run('chat', 'Código fictício TZ-628')
            self.assertIn('TZ-628', response['answer'])

    def test_benchmark_is_deterministic_and_evaluates_answers(self):
        first = run_benchmark(seeds=1, observations=32, queries=8)
        second = run_benchmark(seeds=1, observations=32, queries=8)
        self.assertEqual(first, second)
        self.assertEqual(len(first['summary']), 6)
        self.assertTrue(all(0 <= r['accuracy'] <= 1 for r in first['runs']))


class OptionalBackendsTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec('sentence_transformers') and os.environ.get('LLICAMS_TEST_SENTENCE_MODEL'),
                         'optional sentence encoder/model not configured')
    def test_semantic_backend_encodes_normalized_text(self):
        from llicams.encoders import SentenceTransformerEncoder
        import numpy as np
        encoder = SentenceTransformerEncoder(os.environ['LLICAMS_TEST_SENTENCE_MODEL'])
        vector = encoder.encode('Teste de recuperação de memória celular')
        self.assertAlmostEqual(float(np.linalg.norm(vector)), 1, places=5)
        self.assertEqual(vector.shape, (encoder.dimension,))

    @unittest.skipUnless(importlib.util.find_spec('torch') and importlib.util.find_spec('peft') and importlib.util.find_spec('transformers'),
                         'optional torch/transformers/peft not installed')
    def test_real_tiny_transformer_replay_fusion_and_loading(self):
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast
        from llicams.replay import HFLoRAReplay
        from llicams.llm import TransformersBackend
        with tempfile.TemporaryDirectory() as folder:
            base = str(Path(folder) / 'tiny_base')
            vocabulary = {'[UNK]': 0, '[EOS]': 1, '[PAD]': 2, 'user': 3, 'assistant': 4,
                          'codigo': 5, 'de': 6, 'laboratorio': 7, 'ZX': 8, '821': 9, ':': 10}
            tokenizer = Tokenizer(WordLevel(vocabulary, unk_token='[UNK]'))
            tokenizer.pre_tokenizer = Whitespace()
            wrapped = PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token='[UNK]',
                                              eos_token='[EOS]', pad_token='[PAD]')
            wrapped.save_pretrained(base)
            model = LlamaForCausalLM(LlamaConfig(vocab_size=len(vocabulary), hidden_size=16,
                intermediate_size=32, num_hidden_layers=1, num_attention_heads=2,
                num_key_value_heads=2, max_position_embeddings=128))
            model.save_pretrained(base, safe_serialization=True)
            replay = HFLoRAReplay(base, Path(folder)/'adapters', rank=2, steps=2)
            with Engine(replay=replay) as engine:
                engine.observe('codigo de laboratorio ZX 821', metadata={'pinned': True, 'replay_approved': True})
                first = engine.sleep()
                first_path = engine.stats()['weight_artifact']
                self.assertTrue((Path(first_path)/'adapter_model.safetensors').is_file())
                second = engine.sleep()
                final_path = engine.stats()['weight_artifact']
                self.assertNotEqual(first_path, final_path)
                manifest = json.loads((Path(final_path)/'replay_manifest.json').read_text())
                self.assertTrue(manifest['metrics']['projection'])
                backend = TransformersBackend(base)
                generated = backend.complete([{'role': 'user', 'content': 'codigo de laboratorio'}],
                                             weight_artifact=final_path, max_tokens=3)
                self.assertIsInstance(generated, str)


if __name__ == '__main__':
    unittest.main()
