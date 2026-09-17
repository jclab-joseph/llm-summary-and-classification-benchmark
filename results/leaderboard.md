# LLM Summarization / Hallucination / Classification Benchmark

All models are called through **OpenRouter**. Costs are the authoritative
`usage.cost` values returned by OpenRouter, not pricing-table estimates.

Metrics of different kinds are reported side by side and never merged into a
single composite score.

| Model | API | Input $/M | Output $/M | Summary EN | Summary KO | Halu EN F1 | Halu KO F1 | Classification EN | Classification KO | EN-KO Consistency | Input Tokens | Output Tokens | OpenRouter Cost | Judge Cost |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| google/gemini-3.1-flash-lite | openrouter | 0.2500 | 1.5000 | 0.1800 | 0.2210 | 0.7183 | 0.6113 | 0.8910 | 0.8700 | 0.9330 | 1,098,022 | 55,664 | 0.358001 | 0.000000 |
| qwen/qwen3.8-flash | openrouter | 0.1500 | 0.4700 | 0.1621 | 0.2284 | 0.7597 | 0.5623 | 0.8820 | 0.8460 | 0.9100 | 1,061,596 | 52,179 | 0.147926 | 0.000000 |
| openai/gpt-5.6-luna | openrouter | 0.2000 | 1.2000 | 0.1600 | 0.2296 | 0.7481 | 0.6521 | 0.8680 | 0.8310 | 0.8950 | 1,106,945 | 55,276 | 0.310666 | 0.000000 |
| google/gemini-2.5-flash-lite | openrouter | 0.1000 | 0.4000 | 0.1624 | 0.2287 | 0.6476 | 0.5831 | 0.8430 | 0.8230 | 0.8860 | 1,097,914 | 54,986 | 0.131786 | 0.000000 |
| openai/gpt-5-nano | openrouter | 0.0500 | 0.4000 | 0.1299 | 0.1993 | 0.6815 | 0.3281 | 0.8420 | 0.8150 | 0.8630 | 1,106,945 | 548,399 | 0.272892 | 0.000000 |
| z-ai/glm-5.3-flash | openrouter | 0.1000 | 0.3333 | 0.1443 | 0.2060 | 0.7254 | 0.5556 | 0.8580 | 0.7970 | 0.8670 | 1,293,055 | 119,092 | 0.214717 | 0.000000 |
| deepseek/deepseek-v4-flash-0731 | openrouter | 0.0550 | 0.1100 | 0.1602 | 0.2187 | 0.7204 | 0.6526 | 0.8030 | 0.8020 | 0.8510 | 1,190,712 | 63,519 | 0.087699 | 0.000000 |
| qwen/qwen3-30b-a3b-instruct-2507 | openrouter | 0.0481 | 0.1930 | 0.1416 | 0.2093 | 0.6090 | 0.4191 | 0.8090 | 0.7940 | 0.8650 | 1,213,604 | 72,595 | 0.123234 | 0.000000 |
| qwen/qwen3.7-flash | openrouter | 0.0300 | 0.1300 | 0.1608 | 0.2303 | 0.6590 | 0.5407 | 0.7960 | 0.7600 | 0.8320 | 1,061,596 | 51,608 | 0.037918 | 0.000000 |
| mistralai/ministral-14b-2512 | openrouter | 0.2000 | 0.2000 | 0.1341 | 0.2008 | 0.7163 | 0.6833 | 0.7800 | 0.7440 | 0.8240 | 1,092,316 | 71,944 | 0.179584 | 0.000000 |
| mistralai/ministral-8b-2512 | openrouter | 0.1500 | 0.1500 | 0.1339 | 0.2023 | 0.6366 | 0.5858 | 0.7380 | 0.7370 | 0.7630 | 1,092,316 | 71,698 | 0.134184 | 0.000000 |
| openai/gpt-5.4-nano | openrouter | 0.2000 | 1.2500 | 0.1304 | 0.1873 | 0.6149 | 0.2383 | 0.7150 | 0.6610 | 0.7180 | 1,106,945 | 75,745 | 0.316070 | 0.000000 |
| mistralai/mistral-small-2603 | openrouter | 0.1500 | 0.6000 | 0.1487 | 0.2159 | 0.7248 | 0.6196 | 0.5090 | 0.7110 | 0.7000 | 1,107,916 | 62,923 | 0.196408 | 0.000000 |
| mistralai/ministral-3b-2512 | openrouter | 0.1000 | 0.1000 | 0.1344 | 0.1977 | 0.6757 | 0.5592 | 0.4650 | 0.6060 | 0.6510 | 1,092,316 | 69,602 | 0.087519 | 0.000000 |
| local/qwen2.5-1.5b-instruct-q8-gguf | local | — | — | — | — | — | — | 0.5750 | 0.4620 | 0.7130 | 1,441,986 | 10,903 | — | — |
| local/qwen2.5-1.5b-instruct-q8-gguf-pcd | local | — | — | — | — | — | — | 0.5730 | 0.4570 | 0.7260 | 720,993 | 0 | — | — |
| openai/gpt-4.1-nano | openrouter | 0.1000 | 0.4000 | 0.1434 | 0.2195 | 0.7005 | 0.5422 | 0.5290 | 0.3980 | 0.7310 | 1,108,245 | 55,419 | 0.130275 | 0.000000 |
