# Validation summary 0.1.0

Executed on Python 3.12.14 with NumPy 2.3.5.

- Tests run: 45.
- Passed: 42.
- Failures/errors: 0/0.
- Optional tests skipped: 3; see validation.json.
- Pretrained natural-language LLM performance was not evaluated.
- Tiny random transformer integration test executed: False.
- Linear NumPy LoRA loss: 0.05803241 -> 0.00000000.

## Synthetic delayed-code retrieval

5 seeds, 160 observations and 48 randomly selected queries per seed.
Equal caps of 24 active and 48 shadow items; bytes are recorded but not equalized.
Results are descriptive and do not establish superiority on natural-language tasks.

| Condition | Mean exact-code accuracy | Std across seeds |
|---|---:|---:|
| paper | 0.3167 | 0.0649 |
| paper_no_urgency | 0.3208 | 0.0600 |
| anchored | 0.3042 | 0.0523 |
| recency | 0.4667 | 0.0669 |
| random | 0.4417 | 0.0342 |
| direction_only | 0.0000 | 0.0000 |

No result here demonstrates consciousness or inevitable creative insight.
