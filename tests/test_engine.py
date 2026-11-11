from dataclasses import replace
import json
from pathlib import Path
import tempfile
import threading
import unittest
import numpy as np
from llicams import Config, Engine
from llicams.encoders import HashEncoder
from llicams.index import VectorIndex
from llicams.storage import ConflictError, CapacityError
from llicams.sleep import build_graph
from llicams.llm import ask


class Encoder:
    dimension = 8
    fingerprint = "unit-test-basis-v1"

    def encode(self, text):
        v = np.zeros(8)
        v[int(text.removeprefix("p"))] = 1
        return v


def cfg(**kwargs):
    return Config(dimension=8, topics={str(i): f"p{i}" for i in range(4)}, **kwargs)


ACTIVE = np.array([1, 1, 1, 1, 0, 0, 0, 0])
SHADOW = np.array([1, 1, 1, 0, 0, 0, 0, 0])
LOCAL = np.array([1, 0, 0, 0, 0, 0, 0, 0])


class MemoryTests(unittest.TestCase):
    def setUp(self):
        self.engine = Engine(config=cfg(), encoder=Encoder())

    def tearDown(self):
        self.engine.close()

    def test_pending_visible_before_sleep(self):
        record = self.engine.observe("fact-X71", vector=LOCAL)
        result = self.engine.retrieve(vector=LOCAL)
        self.assertEqual(result["hits"][0].memory.id, record.id)
        self.assertEqual(self.engine.stats()["pending"], 1)

    def test_three_regimes_and_explicit_skipped_replay(self):
        self.engine.observe("active", vector=ACTIVE)
        self.engine.observe("shadow", vector=SHADOW)
        self.engine.observe("discard", vector=LOCAL)
        report = self.engine.sleep()
        stats = self.engine.stats()
        self.assertEqual((stats["active"], stats["shadow"], stats["pending"]), (1, 1, 0))
        self.assertEqual(report["dropped"], 1)
        self.assertEqual(report["phases"][1]["status"], "skipped")

    def test_direction_only_purges_text_and_metadata(self):
        with Engine(config=cfg(shadow_mode="direction_only"), encoder=Encoder()) as e:
            memory = e.observe("SECRET-RANDOM-X919", vector=SHADOW, metadata={"fact_value": "SECRET-RANDOM-X919"})
            e.sleep()
            export = json.dumps(e.export())
            self.assertNotIn("SECRET-RANDOM-X919", export)
            record = e.store.view()["records"][0]
            self.assertIsNone(record.text)
            self.assertEqual(record.metadata, {"pinned": False})
            with self.assertRaises(ValueError):
                e.promote(memory.id, "This still cannot reconstruct the missing fact")

    def test_pointer_shadow_keeps_evidence(self):
        self.engine.observe("fact-X991", vector=SHADOW)
        self.engine.sleep()
        result = self.engine.retrieve(vector=SHADOW)
        self.assertEqual(result["hits"][0].memory.text, "fact-X991")
        self.assertTrue(result["hits"][0].eureka)

    def test_low_urgency_cannot_resurrect(self):
        self.engine.observe("latent", vector=SHADOW)
        self.engine.sleep()
        result = self.engine.retrieve(vector=LOCAL, include_pending=False)
        self.assertLess(result["omega"], self.engine.config.resonance_threshold)
        self.assertEqual(result["hits"], [])

    def test_epsilon_does_not_change_resonance(self):
        self.engine.observe("latent", vector=SHADOW)
        self.engine.sleep()
        before = self.engine.retrieve(vector=SHADOW)["hits"][0].score
        view = self.engine.store.view()
        records = [replace(m, epsilon=1e-9) for m in view["records"]]
        self.engine.store.commit(view["version"], 0, records, [], {"test": "weight independent"})
        after = self.engine.retrieve(vector=SHADOW)["hits"][0].score
        self.assertAlmostEqual(before, after)

    def test_explicit_promotion_preserves_pending(self):
        m = self.engine.observe("latent", vector=SHADOW)
        self.engine.sleep()
        new = self.engine.observe("pending", vector=LOCAL)
        self.engine.promote(m.id, "Caller checked the source")
        states = {m.id: m.state for m in self.engine.store.view()["records"]}
        self.assertEqual(states[m.id], "active")
        self.assertEqual(states[new.id], "pending")

    def test_pin_and_graph_degree_cap(self):
        with Engine(config=cfg(max_active=3, max_shadows=2, max_degree=1, cluster_similarity=1.0), encoder=Encoder()) as e:
            for i in range(7):
                v = ACTIVE + np.random.default_rng(i).normal(0, 0.05, 8)
                e.observe(f"fact-{i}", vector=v, metadata={"pinned": i == 0}, memory_id=str(i))
            e.sleep()
            view = e.store.view()
            self.assertEqual(e.stats()["active"], 3)
            self.assertLessEqual(e.stats()["shadow"], 2)
            self.assertEqual(next(m for m in view["records"] if m.id == "0").state, "active")
            degrees = {}
            for a, b, _ in view["edges"]:
                self.assertNotEqual(a, b)
                degrees[a] = degrees.get(a, 0) + 1
                degrees[b] = degrees.get(b, 0) + 1
            self.assertTrue(all(d <= 1 for d in degrees.values()))

    def test_pending_and_payload_limits(self):
        with Engine(config=cfg(max_pending=1, max_total_text_bytes=4), encoder=Encoder()) as e:
            e.observe("abc", vector=LOCAL)
            with self.assertRaises(CapacityError):
                e.observe("d", vector=LOCAL)
        with Engine(config=cfg(max_total_text_bytes=4), encoder=Encoder()) as e:
            with self.assertRaises(CapacityError):
                e.observe("abcde", vector=ACTIVE)

    def test_identity_checks_on_reopen(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "m.sqlite"
            with Engine(path, cfg(), Encoder()) as e:
                e.observe("stable", vector=ACTIVE)
                e.sleep()
            with Engine(path, cfg(), Encoder()) as e:
                self.assertEqual(e.stats()["active"], 1)
            with self.assertRaises(ValueError):
                Engine(path, cfg(lambda_li=0.9), Encoder())

    def test_paper_score_can_select_uniform_noise(self):
        noise = np.array([0, 0, 0, 0, 1, 0, 0, 0])
        self.engine.observe("unanchored", vector=noise)
        self.engine.sleep()
        self.assertEqual(self.engine.stats()["active"], 1)
        with Engine(config=cfg(selector="anchored"), encoder=Encoder()) as e:
            e.observe("unanchored", vector=noise)
            e.sleep()
            self.assertEqual(e.stats()["active"], 0)

    def test_invalid_inputs(self):
        for text, kwargs in [("", {}), ("x", {"vector": np.zeros(8)}),
                             ("x", {"vector": ACTIVE, "confidence": float("nan")}),
                             ("x", {"vector": ACTIVE, "metadata": {"pinned": "yes"}})]:
            with self.assertRaises(ValueError):
                self.engine.observe(text, **kwargs)

    def test_endogenous_is_read_only(self):
        for i in range(3):
            self.engine.observe(f"pinned-{i}", vector=ACTIVE + np.eye(8)[i] * .1,
                                metadata={"pinned": True})
        self.engine.sleep()
        before = self.engine.export()
        proposals = self.engine.endogenous()
        self.assertGreater(len(proposals), 0)
        self.assertEqual(before, self.engine.export())

    def test_structural_rem_conflicts_are_unverified(self):
        for value in ("A", "B"):
            self.engine.observe(value, vector=ACTIVE, metadata={"pinned": True,
                                "fact_key": "location", "fact_value": value})
        self.engine.sleep()
        proposals = self.engine.store.proposals()
        self.assertTrue(any(p["explicit_conflict"] for p in proposals))
        self.assertTrue(all(p["status"] == "unverified" for p in proposals))

    def test_direction_only_is_not_fabricated_into_context(self):
        with Engine(config=cfg(shadow_mode="direction_only"), encoder=Encoder()) as e:
            e.observe("secret code ABC921", vector=SHADOW)
            e.sleep()
            # Override query encoder for a controlled matching query.
            e.encoder.encode = lambda _: SHADOW
            answer = ask(e, "recover code")
            self.assertNotIn("ABC921", answer["answer"])


class ConcurrencyTests(unittest.TestCase):
    def test_observation_during_sleep_is_not_lost(self):
        entered, release = threading.Event(), threading.Event()
        class Replay:
            def prepare(self, records, previous):
                entered.set()
                if not release.wait(4):
                    raise RuntimeError("test timeout")
                return {"artifact": previous, "metrics": {"status": "skipped"}}
        with Engine(config=cfg(), encoder=Encoder()) as e:
            e.observe("old", vector=ACTIVE)
            e.sleep()
            e.replay = Replay()
            captured = e.observe("captured", vector=ACTIVE)
            errors = []
            def worker():
                try: e.sleep()
                except Exception as exc: errors.append(exc)
            thread = threading.Thread(target=worker)
            thread.start()
            try:
                self.assertTrue(entered.wait(2))
                late = e.observe("arrived during sleep", vector=LOCAL)
                self.assertEqual(e.retrieve(vector=ACTIVE)["version"], 1)
            finally:
                release.set()
                thread.join(5)
            self.assertFalse(errors)
            self.assertFalse(thread.is_alive())
            states = {m.id: m.state for m in e.store.view()["records"]}
            self.assertEqual(states[late.id], "pending")
            self.assertIn(states[captured.id], {"active", "shadow"})

    def test_preparation_failure_leaves_state_unchanged(self):
        class BrokenReplay:
            def prepare(self, records, previous):
                raise RuntimeError("intentional failure")
        with Engine(config=cfg(), encoder=Encoder(), replay=BrokenReplay()) as e:
            e.observe("pending", vector=ACTIVE)
            before = e.export()
            with self.assertRaises(RuntimeError): e.sleep()
            self.assertEqual(before, e.export())

    def test_stale_candidate_cannot_overwrite_other_engine(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "state.sqlite"
            with Engine(path, cfg(), Encoder()) as a, Engine(path, cfg(), Encoder()) as b:
                m = a.observe("shadow", vector=SHADOW)
                a.sleep()
                stale = a.store.view()
                b.promote(m.id, "new source evidence")
                with self.assertRaises(ConflictError):
                    a.store.commit(stale["version"], stale["seq"], [], [], {"stale": True})
                self.assertEqual(a.stats()["active"], 1)


class EncoderAndIndexTests(unittest.TestCase):
    def test_stable_hash_and_accents(self):
        encoder = HashEncoder(64, 5)
        np.testing.assert_allclose(encoder.encode("Memória celular"), encoder.encode("memoria celular"))
        np.testing.assert_allclose(encoder.encode("x y"), HashEncoder(64, 5).encode("x y"))

    def test_exact_index_order(self):
        with Engine(config=cfg(), encoder=Encoder()) as e:
            a = e.observe("a", vector=LOCAL)
            b = e.observe("b", vector=ACTIVE)
            matches = VectorIndex(e.store.view()["records"]).search(LOCAL, 50)
            self.assertEqual(matches[0][0].id, a.id)
            self.assertEqual(len(matches), 2)

    @unittest.skipUnless(__import__('importlib.util').util.find_spec('hnswlib'), 'optional hnswlib not installed')
    def test_hnsw_matches_exact_for_known_nearest(self):
        with Engine(config=cfg(), encoder=Encoder()) as e:
            for i in range(20):
                e.observe(str(i), vector=np.random.default_rng(i).normal(size=8))
            records = e.store.view()["records"]
            index = VectorIndex(records, "hnsw")
            for m in records:
                self.assertEqual(index.search(m.vector, 1)[0][0].id, m.id)
