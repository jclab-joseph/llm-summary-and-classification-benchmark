# LLM Summarization / Hallucination / Classification Benchmark

All models are called through **OpenRouter**. Costs are the authoritative
`usage.cost` values returned by OpenRouter, not pricing-table estimates.

Metrics of different kinds are reported side by side and never merged into a
single composite score.

| Model | Input $/M | Output $/M | Summary EN | Summary KO | Halu EN F1 | Halu KO F1 | Classification EN | Classification KO | EN-KO Consistency | Input Tokens | Output Tokens | OpenRouter Cost | Judge Cost |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| google/gemini-2.5-flash-lite | 0.1000 | 0.4000 | 0.1401 | 0.2298 | 0.4604 | 0.5353 | 0.0080 | 0.0130 | 0.9790 | 844,524 | 24,718 | 0.094340 | 0.000000 |
