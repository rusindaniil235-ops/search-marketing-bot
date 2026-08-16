# Evaluation report

- Canonical cases: 46; accuracy: 100.0%
- Holdout cases: 24; accuracy: 95.8%
- Overall cases: 70; accuracy: 98.6%
- Median classification latency: 0.1610 ms
- P95 classification latency: 0.1880 ms

## Per-intent accuracy

| Intent | Correct | Total | Accuracy |
|---|---:|---:|---:|
| anomaly | 2 | 2 | 100.0% |
| attribution | 1 | 2 | 50.0% |
| conversion | 2 | 2 | 100.0% |
| data_quality | 2 | 2 | 100.0% |
| experiment | 2 | 2 | 100.0% |
| fallback | 2 | 2 | 100.0% |
| greeting | 2 | 2 | 100.0% |
| help | 2 | 2 | 100.0% |
| reporting | 2 | 2 | 100.0% |
| search_reach | 2 | 2 | 100.0% |
| team_scope | 2 | 2 | 100.0% |
| traffic_sources | 2 | 2 | 100.0% |

## Errors

- `Трафик упал после смены модели атрибуции`: expected `attribution`, got `anomaly`
