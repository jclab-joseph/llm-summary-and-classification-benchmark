# 벤치마크 결과

<!-- 이 파일은 `benchmark report` 가 자동 생성합니다. 직접 수정하지 마세요. -->

생성: `2026-09-17T00:38:53+00:00` · benchmark version `1.0.0` · API provider: **OpenRouter**

같은 데이터를 담은 스프레드시트: [`results/benchmark_results.xlsx`](results/benchmark_results.xlsx)

## 벤치마크 구성

세 가지 능력을 영어와 한국어에서 각각 측정합니다. 모델 하나당 3,200 케이스,
OpenRouter 요청 1,300건입니다.

| 벤치마크 | 데이터셋 | 과제 | 케이스 |
|---|---|---|---:|
| 요약 | XL-Sum (BBC 뉴스) | 기사를 1~3문장으로 요약 | EN 200 + KO 200 |
| 환각 탐지 | HaluEval (EN) / AI-Hub 추상요약 사실성 검증 (KO) | 요약이 원문에 의해 뒷받침되는지 `SUPPORTED`/`HALLUCINATED` 판정 | EN 400 + KO 400 |
| 분류 | MASSIVE | 발화를 60개 인텐트 중 하나로 분류 (20건씩 묶어 요청) | EN 1,000 + KO 1,000 |

모든 모델이 동일한 frozen manifest(같은 샘플, 같은 전처리, 같은 프롬프트)를 받습니다.

## 숫자를 읽기 전에

각 표 아래 `지표 설명` 을 펼치면 지표의 의미와 방향이 나옵니다 (↑ 높을수록 좋음 · ↓ 낮을수록 좋음 · — 좋고 나쁨이 아니라 해석용).

- 비용은 전부 OpenRouter가 반환한 **실제 청구액 `usage.cost`** 입니다. 가격표 추정치가 아닙니다.
- 성격이 다른 지표를 하나의 종합 점수로 합치지 않습니다. 열끼리 따로 보세요.
- **Surface fact support 는 진단 지표** 이지 환각 지표가 아닙니다. 표면적 사실만 대조합니다.
- EN/KO 환각 점수는 출처가 달라(HaluEval vs AI-Hub) 서로 직접 비교 대상이 아닙니다.
- 한국어 ROUGE 는 문자 단위 토큰화라 영어 ROUGE 와 직접 비교할 수 없습니다.
- 빈 칸(`—`)은 값이 0이라는 뜻이 아니라 **계산되지 않았다**는 뜻입니다.

## 리더보드

각 벤치마크의 대표 지표만 모은 표입니다. `Summary` 는 ROUGE-Lsum, `Halu` 는 HALLUCINATED F1, `Cls` 는 정확도, `EN-KO` 는 교차 일관성입니다. 자세한 값은 아래 벤치마크별 표를 보세요. 정렬 기준은 분류 전체 정확도입니다.

| Model | Input $/M | Output $/M | Summary EN | Summary KO | Halu EN F1 | Halu KO F1 | Cls EN | Cls KO | EN-KO | Input Tokens | Output Tokens | OpenRouter Cost | Judge Cost |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| google/gemini-2.5-flash-lite | $0.1000 | $0.4000 | 0.1401 | 0.2298 | 0.4604 | 0.5353 | 0.0080 | 0.0130 | 0.9790 | 844,524 | 24,718 | $0.094340 | $0.000000 |

<details><summary>지표 설명</summary>

| 지표 | 방향 | 의미 |
| --- | :--: | --- |
| Input $/M, Output $/M | ↓ | 100만 토큰당 가격(USD). 실행 시점 OpenRouter 가격 스냅샷. |
| Summary EN / KO | ↑ | 요약 ROUGE-Lsum F1 (0~1). |
| Halu EN / KO F1 | ↑ | 환각 탐지에서 HALLUCINATED 를 양성으로 본 F1 (0~1). |
| Cls EN / KO | ↑ | 분류 정확도 (0~1). 무작위는 약 0.017. |
| EN-KO | ↑ | 두 언어에서 정오답이 일치한 비율 (0~1). |
| Input / Output Tokens | — | 누적 토큰 수. |
| OpenRouter Cost | ↓ | 누적 실제 청구액. |
| Judge Cost | ↓ | judge 사용 시의 청구액. 기본 예산과 별도. |

</details>

## 요약 (XL-Sum)

XL-Sum(BBC 뉴스)의 기사를 1~3문장으로 요약시키고, 기사에 딸린 사람이 쓴
참조 요약과 비교합니다. 원문은 약 1,800토큰으로 잘라 모든 모델에 동일하게 제공합니다.

| Model | Lang | Cases | ROUGE-Lsum | chrF++ | BERTScore | Surface facts* | Compression | Out tokens | Empty | Refusal |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| google/gemini-2.5-flash-lite | en | 200 | 0.1401 | 22.50 | — | 1.0000 | 0.1316 | 51.70 | 0.0000 | 0.0000 |
| google/gemini-2.5-flash-lite | ko | 200 | 0.2298 | 12.43 | — | 1.0000 | 0.1147 | 71.78 | 0.0000 | 0.0000 |

<details><summary>지표 설명</summary>

| 지표 | 방향 | 의미 |
| --- | :--: | --- |
| Cases | — | 채점 대상 케이스 수. |
| ROUGE-Lsum | ↑ | 참조 요약과의 최장 공통 부분열 기반 F1 (0~1). 문장 단위로 계산. 한국어는 문자 단위 토큰화라 영어 값과 직접 비교 불가. |
| chrF++ | ↑ | 문자 n-gram + 단어 bigram F-score (0~100). 어형 변화에 강해 한국어에 비교적 공정. |
| BERTScore | ↑ | 다국어 임베딩 기반 의미 유사도 F1 (0~1). 표현이 달라도 뜻이 같으면 점수를 줍니다. 선택 설치. |
| Surface facts | ↑ | 요약에서 뽑은 표면적 사실 중 원문에서 확인되는 비율 (0~1). 진단 지표. |
| Compression | — | 출력 길이 ÷ 원문 길이. 낮으면 압축적, 높으면 장황. 낮다고 무조건 좋은 것은 아닙니다. |
| Out tokens | — | 평균 출력 길이(추정 토큰). |
| Empty | ↓ | 빈 출력 비율. |
| Refusal | ↓ | 거부 응답 비율. 빈 출력과 거부는 분모에서 빼지 않습니다. |

</details>

`*` Surface facts 는 진단 지표입니다.

## 환각 탐지

원문과 후보 요약을 함께 주고 요약의 모든 사실적 주장이 원문에 의해
뒷받침되는지 판단시킵니다. 문서 하나당 두 케이스(충실한 요약 = `SUPPORTED`, 오류가 있는 요약 =
`HALLUCINATED`)라 레이블이 정확히 반반입니다. 따라서 **무작위 추측의 정확도는 0.5** 입니다.

| Model | Lang | Status | Cases | Accuracy | Macro-F1 | Halu P | Halu R | Halu F1 | Supported R | Invalid |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| google/gemini-2.5-flash-lite | en | OK | 400 | 0.4725 | 0.4722 | 0.4712 | 0.4500 | 0.4604 | 0.4950 | 0.0000 |
| google/gemini-2.5-flash-lite | ko | OK | 400 | 0.5225 | 0.5221 | 0.5213 | 0.5500 | 0.5353 | 0.4950 | 0.0000 |

<details><summary>지표 설명</summary>

| 지표 | 방향 | 의미 |
| --- | :--: | --- |
| Status | — | `OK` 또는 `SKIPPED`(해당 언어 데이터 미준비). |
| Cases | — | 케이스 수. 문서 수 × 2. |
| Accuracy | ↑ | 전체 정답률 (0~1). 레이블이 반반이므로 0.5가 무작위 수준. |
| Macro-F1 | ↑ | 두 레이블 F1의 단순 평균 (0~1). 한쪽 레이블만 찍는 모델을 걸러냅니다. |
| Halu P | ↑ | HALLUCINATED 라고 답한 것 중 실제로 맞은 비율. 낮으면 멀쩡한 요약을 환각이라고 의심. |
| Halu R | ↑ | 실제 HALLUCINATED 중 잡아낸 비율. 낮으면 환각을 놓침. |
| Halu F1 | ↑ | Halu P 와 Halu R 의 조화평균. 환각 탐지 성능의 대표값. |
| Supported R | ↑ | 실제 SUPPORTED 중 맞힌 비율. 이 값만 낮으면 과도하게 의심하는 모델. |
| Invalid | ↓ | 레이블로 파싱할 수 없는 출력 비율. **오답으로 계산됩니다.** |

</details>

한국어는 AI-Hub 데이터가 준비되지 않으면 `SKIPPED` 로 표시됩니다.

## 분류 (MASSIVE)

MASSIVE 음성비서 발화를 60개 인텐트 중 하나로 분류시킵니다. 비용 절감을
위해 20건씩 묶어 한 요청으로 보내고, `<번호>: <인텐트 번호>` 형식으로 답하게 합니다.
**무작위 추측의 정확도는 1/60 ≈ 0.017** 입니다.

| Model | Scope | Cases | Accuracy | Macro-F1 | Invalid |
| --- | --- | ---: | ---: | ---: | ---: |
| google/gemini-2.5-flash-lite | en | 1,000 | 0.0080 | 0.0081 | 0.0000 |
| google/gemini-2.5-flash-lite | ko | 1,000 | 0.0130 | 0.0121 | 0.0000 |
| google/gemini-2.5-flash-lite | overall | 2,000 | 0.0105 | 0.0099 | 0.0000 |

<details><summary>지표 설명</summary>

| 지표 | 방향 | 의미 |
| --- | :--: | --- |
| Scope | — | `en` / `ko` / `overall`(두 언어 합산). |
| Cases | — | 분류한 발화 수. |
| Accuracy | ↑ | 정답률 (0~1). |
| Macro-F1 | ↑ | 60개 인텐트별 F1의 평균 (0~1). 드문 인텐트를 무시하는 모델에 불리하게 작동합니다. |
| Invalid | ↓ | 형식을 못 지켜 답을 못 읽은 비율. 오답으로 계산됩니다. 배치 형식을 못 지키면 여기가 올라갑니다. |

</details>

## EN/KO 교차 일관성

분류 벤치마크의 EN/KO 쌍은 **같은 semantic id** 를 공유합니다. 즉 뜻이 같은
발화를 두 언어로 물어본 것이라, 두 언어 사이의 성능 편차를 직접 볼 수 있습니다.

| Model | Pairs | Consistency | Pred. agreement | Both correct | EN only | KO only | Both wrong |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| google/gemini-2.5-flash-lite | 1,000 | 0.9790 | 0.0210 | 0.0000 | 0.0080 | 0.0130 | 0.9790 |

<details><summary>지표 설명</summary>

| 지표 | 방향 | 의미 |
| --- | :--: | --- |
| Pairs | — | 비교한 EN/KO 쌍의 수. |
| Consistency | ↑ | 두 언어에서 정오답이 일치한 비율(둘 다 맞음 + 둘 다 틀림). 높을수록 언어에 따라 흔들리지 않음. |
| Pred. agreement | ↑ | 정답 여부와 무관하게 **같은 인텐트** 를 예측한 비율. 정확도와 분리된 안정성 지표. |
| Both correct | ↑ | 두 언어 모두 맞힌 비율. |
| EN only | ↓ | 영어만 맞힌 비율. 높으면 한국어가 약함. |
| KO only | ↓ | 한국어만 맞힌 비율. 높으면 영어가 약함. |
| Both wrong | ↓ | 두 언어 모두 틀린 비율. |

</details>

## Surface fact support (진단)

생성된 요약에서 숫자·백분율·날짜·통화·개체명을 규칙 기반으로 추출해 원문과
대조합니다. **LLM 을 쓰지 않는 결정적 진단** 이며, 완전한 환각 지표가 아닙니다. 의미 수준의
왜곡("부인했다" vs "확인했다")이나 누락은 잡지 못하고, 개체명은 정밀도 위주로 적게 뽑습니다.

| Model | Lang | Facts | Support | Number | Percent | Date | Currency | Entity |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| google/gemini-2.5-flash-lite | en | 1,031 | 1.0000 | 0 | 0 | 0 | 0 | 0 |
| google/gemini-2.5-flash-lite | ko | 642 | 1.0000 | 0 | 0 | 0 | 0 | 0 |

<details><summary>지표 설명</summary>

| 지표 | 방향 | 의미 |
| --- | :--: | --- |
| Facts | — | 요약에서 추출된 사실의 총 개수. 적으면 아래 비율의 신뢰도가 낮습니다. |
| Support | ↑ | 그중 원문에서 확인된 비율 (0~1). |
| Number | ↓ | 원문에 없는 숫자 건수. |
| Percent | ↓ | 원문과 다른 백분율 건수. (원문 12% → 요약 21% 같은 경우) |
| Date | ↓ | 원문에 없는 날짜 건수. |
| Currency | ↓ | 원문에 없는 금액 건수. |
| Entity | ↓ | 원문에 없는 고유명사 건수. |

</details>

## 비용 (OpenRouter usage.cost)

모두 OpenRouter 가 응답에 실어 보낸 **실제 청구액 `usage.cost`** 입니다. 가격표로
다시 계산한 값이 아닙니다.

| Model | Summarization | Hallucination | Classification | Retry/error | Judge | Total | Fresh this run | Cache savings |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| google/gemini-2.5-flash-lite | $0.031904 | $0.056225 | $0.006210 | $0.000000 | $0.000000 | $0.094340 | $0.000000 | $0.094340 |

<details><summary>지표 설명</summary>

| 지표 | 방향 | 의미 |
| --- | :--: | --- |
| Summarization / Hallucination / Classification | ↓ | 벤치마크별 누적 청구액. |
| Retry/error | ↓ | 실패했지만 과금된 시도의 합. 0이 아니면 재시도가 돈을 쓴 것. |
| Judge | ↓ | LLM-as-a-Judge 비용. 기본 예산과 별도로 집계합니다. |
| Total | ↓ | 이 모델 캐시를 만드는 데 지금까지 지불한 총액. |
| Fresh this run | — | 마지막 실행에서 **새로** 청구된 금액. 캐시가 다 맞으면 $0. |
| Cache savings | ↑ | 캐시 덕분에 다시 내지 않은 금액. |

</details>

## 토큰 사용량 / 캐시

토큰 수는 OpenRouter 가 보고한 값입니다. 같은 케이스를 푸는 데 토큰을 적게 쓰면
싸지만, 출력 토큰이 지나치게 적으면 요약이 잘렸다는 뜻일 수도 있습니다.

| Model | Cases | Prompt | Cached | Completion | Reasoning | Total | API calls | Cache hits |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| google/gemini-2.5-flash-lite | 1,300 | 844,524 | 0 | 24,718 | 0 | 869,242 | 0 | 1,300 |

<details><summary>지표 설명</summary>

| 지표 | 방향 | 의미 |
| --- | :--: | --- |
| Cases | — | 성공적으로 저장된 케이스 수. |
| Prompt / Completion / Total | — | 입력·출력·합계 토큰. |
| Cached | ↑ | 프롬프트 캐시로 할인된 입력 토큰. |
| Reasoning | — | thinking 토큰. thinking 을 끌 수 없는 모델만 0이 아닙니다. 과금 대상입니다. |
| API calls | ↓ | 실제로 OpenRouter 로 나간 요청 수. 재실행에서 0이면 캐시가 완전히 맞은 것. |
| Cache hits | ↑ | 캐시로 처리해 요청하지 않은 수. |

</details>

## 재현성

결과를 재현하는 데 필요한 정보입니다. **manifest 해시가 모든 모델에서 같으면**
완전히 동일한 케이스로 비교했다는 뜻입니다.

| Model | Status | 실행 시각(UTC) | Bench ver. | Git commit | Cases | XL-Sum | HaluEval | AI-Hub | MASSIVE |
| --- | --- | --- | --- | --- | ---: | --- | --- | --- | --- |
| google/gemini-2.5-flash-lite | COMPLETED | 2026-09-17T00:38:50+00:00 | 1.0.0 | 259eab707098 | 3,200 | 0135c4f4499e | 21debf696ed0 | b49d914db700 | 1bebd7a4e6ca |

