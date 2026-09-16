# 벤치마크 결과

<!-- 이 파일은 `benchmark report` 가 자동 생성합니다. 직접 수정하지 마세요. -->

생성: `2026-09-16T04:09:51+00:00` · benchmark version `1.0.0` · API provider: **OpenRouter**

같은 데이터를 담은 스프레드시트: [`results/benchmark_results.xlsx`](results/benchmark_results.xlsx)

## 숫자를 읽기 전에

- 비용은 전부 OpenRouter가 반환한 **실제 청구액 `usage.cost`** 입니다. 가격표 추정치가 아닙니다.
- 성격이 다른 지표를 하나의 종합 점수로 합치지 않습니다. 열끼리 따로 보세요.
- **Surface fact support 는 진단 지표** 이지 환각 지표가 아닙니다. 표면적 사실만 대조합니다.
- EN/KO 환각 점수는 출처가 달라(HaluEval vs AI-Hub) 서로 직접 비교 대상이 아닙니다.
- 한국어 ROUGE 는 문자 단위 토큰화라 영어 ROUGE 와 직접 비교할 수 없습니다.
- 빈 칸(`—`)은 값이 0이라는 뜻이 아니라 **계산되지 않았다**는 뜻입니다.

## 리더보드

| Model | Input $/M | Output $/M | Summary EN | Summary KO | Halu EN F1 | Halu KO F1 | Cls EN | Cls KO | EN-KO | Input Tokens | Output Tokens | OpenRouter Cost | Judge Cost |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| google/gemini-2.5-flash-lite | $0.1000 | $0.4000 | 0.1401 | 0.2298 | 0.4604 | 0.5353 | 0.0080 | 0.0130 | 0.9790 | 844,524 | 24,718 | $0.094340 | $0.000000 |

## 요약 (XL-Sum)

`*` Surface facts 는 진단 지표입니다.

| Model | Lang | Cases | ROUGE-Lsum | chrF++ | BERTScore | Surface facts* | Compression | Out tokens | Empty | Refusal |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| google/gemini-2.5-flash-lite | en | 200 | 0.1401 | 22.50 | — | 1.0000 | 0.1316 | 51.70 | 0.0000 | 0.0000 |
| google/gemini-2.5-flash-lite | ko | 200 | 0.2298 | 12.43 | — | 1.0000 | 0.1147 | 71.78 | 0.0000 | 0.0000 |

## 환각 탐지

한국어는 AI-Hub 데이터가 준비되지 않으면 `SKIPPED` 로 표시됩니다.

| Model | Lang | Status | Cases | Accuracy | Macro-F1 | Halu P | Halu R | Halu F1 | Supported R | Invalid |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| google/gemini-2.5-flash-lite | en | OK | 400 | 0.4725 | 0.4722 | 0.4712 | 0.4500 | 0.4604 | 0.4950 | 0.0000 |
| google/gemini-2.5-flash-lite | ko | OK | 400 | 0.5225 | 0.5221 | 0.5213 | 0.5500 | 0.5353 | 0.4950 | 0.0000 |

## 분류 (MASSIVE)

| Model | Scope | Cases | Accuracy | Macro-F1 | Invalid |
| --- | --- | ---: | ---: | ---: | ---: |
| google/gemini-2.5-flash-lite | en | 1,000 | 0.0080 | 0.0081 | 0.0000 |
| google/gemini-2.5-flash-lite | ko | 1,000 | 0.0130 | 0.0121 | 0.0000 |
| google/gemini-2.5-flash-lite | overall | 2,000 | 0.0105 | 0.0099 | 0.0000 |

## EN/KO 교차 일관성

같은 semantic id 를 공유하는 EN/KO 쌍 기준입니다.

| Model | Pairs | Consistency | Pred. agreement | Both correct | EN only | KO only | Both wrong |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| google/gemini-2.5-flash-lite | 1,000 | 0.9790 | 0.0210 | 0.0000 | 0.0080 | 0.0130 | 0.9790 |

## Surface fact support (진단)

숫자·백분율·날짜·통화·개체명을 원문과 대조한 결과입니다. 완전한 환각 지표가 아닙니다.

| Model | Lang | Facts | Support | Number | Percent | Date | Currency | Entity |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| google/gemini-2.5-flash-lite | en | 1,031 | 1.0000 | 0 | 0 | 0 | 0 | 0 |
| google/gemini-2.5-flash-lite | ko | 642 | 1.0000 | 0 | 0 | 0 | 0 | 0 |

## 비용 (OpenRouter usage.cost)

`Fresh this run` 은 마지막 실행에서 새로 청구된 금액, `Total` 은 이 모델 캐시를 만드는 데 지금까지 지불한 총액입니다.

| Model | Summarization | Hallucination | Classification | Retry/error | Judge | Total | Fresh this run | Cache savings |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| google/gemini-2.5-flash-lite | $0.031904 | $0.056225 | $0.006210 | $0.000000 | $0.000000 | $0.094340 | $0.000000 | $0.094340 |

## 토큰 사용량 / 캐시

| Model | Cases | Prompt | Cached | Completion | Reasoning | Total | API calls | Cache hits |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| google/gemini-2.5-flash-lite | 1,300 | 844,524 | 0 | 24,718 | 0 | 869,242 | 0 | 1,300 |

## 재현성

manifest 해시가 같으면 모든 모델이 완전히 동일한 케이스를 받았다는 뜻입니다.

| Model | Status | 실행 시각(UTC) | Bench ver. | Git commit | Cases | XL-Sum | HaluEval | AI-Hub | MASSIVE |
| --- | --- | --- | --- | --- | ---: | --- | --- | --- | --- |
| google/gemini-2.5-flash-lite | COMPLETED | 2026-09-16T04:09:49+00:00 | 1.0.0 | 1a5b74b324b5 | 3,200 | 0135c4f4499e | 21debf696ed0 | b49d914db700 | 1bebd7a4e6ca |

