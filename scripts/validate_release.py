"""Execute validation and write factual machine-readable release records."""
import importlib.metadata
import io
import json
from pathlib import Path
import platform
import py_compile
import sys
import unittest
from datetime import datetime, timezone
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from llicams.benchmark import run_benchmark
from llicams.math import fit_linear_lora, project_orthogonal, rank_compress


def write(name, data):
    path = ROOT / 'validation' / name
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def main():
    for directory in ('src', 'tests', 'examples', 'scripts'):
        for file in (ROOT / directory).rglob('*.py'):
            py_compile.compile(str(file), doraise=True)
    stream = io.StringIO()
    suite = unittest.defaultTestLoader.discover(str(ROOT / 'tests'))
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    (ROOT / 'validation').mkdir(exist_ok=True)
    (ROOT / 'validation/unittest.txt').write_text(stream.getvalue(), encoding='utf-8')
    dependencies = {}
    for name in ('numpy', 'torch', 'transformers', 'peft', 'sentence-transformers', 'hnswlib'):
        try: dependencies[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: dependencies[name] = None
    skipped_names = [str(t) for t, _ in result.skipped]
    tiny_executed = dependencies['torch'] is not None and not any('test_real_tiny_transformer' in t for t in skipped_names)
    report = {'software': 'llicams-reference', 'version': '0.1.0', 'release_date': '2026-11-11',
              'run_utc': datetime.now(timezone.utc).isoformat(),
              'python': platform.python_version(), 'platform': platform.system(), 'dependencies': dependencies,
              'tests_run': result.testsRun,
              'passed': result.testsRun - len(result.errors) - len(result.failures) - len(result.skipped),
              'failures': len(result.failures), 'errors': len(result.errors),
              'skipped': [{'test': str(t), 'reason': reason} for t, reason in result.skipped],
              'success': result.wasSuccessful(),
              'tiny_random_transformer_training_executed': tiny_executed,
              'pretrained_language_model_performance_evaluated': False,
              'ollama_real_model_executed': False,
              'browser_visual_check': 'not completed in delivery environment; HTTP/application routes were executed'}
    write('validation.json', report)
    if not result.wasSuccessful():
        print(stream.getvalue())
        return 1
    benchmark = run_benchmark(seeds=5, observations=160, queries=48)
    write('benchmark.json', benchmark)
    rng = np.random.default_rng(5)
    x = rng.normal(size=(80,4)); y = x + x @ np.diag([.4, -.3, 0, 0])
    b, a, losses = fit_linear_lora(x, y, rank=2, steps=1500, learning_rate=.3)
    projected = project_orthogonal(rng.normal(size=(4,4)), [b @ a])
    _, _, residual = rank_compress(b @ a + projected, 2)
    linear = {'experiment': 'linear_numpy_lora_not_transformer', 'first_loss': losses[0],
              'last_loss': losses[-1], 'orthogonal_inner_product': float(np.sum(projected * (b @ a))),
              'recompression_residual': residual}
    write('linear_lora.json', linear)
    rows = '\n'.join(f"| {r['condition']} | {r['mean_accuracy']:.4f} | {r['std_across_seeds']:.4f} |"
                     for r in benchmark['summary'])
    summary = f'''# Validation summary 0.1.0

Executed on Python {report['python']} with NumPy {dependencies['numpy']}.

- Tests run: {result.testsRun}.
- Passed: {report['passed']}.
- Failures/errors: {len(result.failures)}/{len(result.errors)}.
- Optional tests skipped: {len(result.skipped)}; see validation.json.
- Pretrained natural-language LLM performance was not evaluated.
- Tiny random transformer integration test executed: {tiny_executed}.
- Linear NumPy LoRA loss: {losses[0]:.8f} -> {losses[-1]:.8f}.

## Synthetic delayed-code retrieval

5 seeds, 160 observations and 48 randomly selected queries per seed.
Equal caps of 24 active and 48 shadow items; bytes are recorded but not equalized.
Results are descriptive and do not establish superiority on natural-language tasks.

| Condition | Mean exact-code accuracy | Std across seeds |
|---|---:|---:|
{rows}

No result here demonstrates consciousness or inevitable creative insight.
'''
    (ROOT / 'validation/SUMMARY.md').write_text(summary, encoding='utf-8')
    (ROOT / 'validation/requirements-tested.txt').write_text(f"# Python {report['python']}\nnumpy=={dependencies['numpy']}\n")
    print(json.dumps({'tests': report['tests_run'], 'passed': report['passed'],
                      'skipped': len(result.skipped), 'benchmark': benchmark['summary'],
                      'linear_lora': linear}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
