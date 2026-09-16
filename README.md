# LLM 요약 / 환각 / 분류 벤치마크

저비용 LLM을 **OpenRouter 하나만으로** 평가하는, 재현 가능한 소형 벤치마크입니다.
고정된 데이터셋에서 **영어와 한국어** 세 가지 능력을 측정하며, 새 모델이 나왔을 때
YAML 한 블록만 추가하면 바로 비교할 수 있도록 만들었습니다.

```bash
uv sync
cp .env.example .env            # OpenRouter 키 입력
benchmark prepare
benchmark estimate --model google/gemini-2.5-flash-lite
benchmark run      --model google/gemini-2.5-flash-lite
benchmark report
```

---

## 왜 만들었나

저렴한 모델은 몇 주 단위로 바뀝니다. "새로 나온 Flash-Lite가 지금 쓰는 것보다 실제로 나은가?"
라는 질문에 답하는 데 오후 반나절과 즉석 스크립트가 아니라, 10분과 1달러 미만이면 충분해야 합니다.

이를 가능하게 하는 세 가지 성질을 중심으로 설계했습니다.

1. **Frozen manifest.** 샘플 선정은 딱 한 번만 일어납니다. 모든 모델이 같은 데이터셋 리비전에서,
   같은 방식으로 잘린, 같은 질문을 받습니다.
2. **캐시 우선 설계.** 한 번 성공한 추론은 다시 요청하지 않습니다. 끝난 벤치마크를 다시 돌리면
   비용은 정확히 `$0.0000` 입니다.
3. **실제 비용 기준.** 비용 리포트의 모든 숫자는 가격표 추정치가 아니라 OpenRouter가 반환한
   `usage.cost` 입니다.

---

## 무엇을 측정하는가

| 벤치마크 | 데이터셋 | 케이스 | 요청 수 |
|---|---|---:|---:|
| 요약 | XL-Sum (`csebuetnlp/xlsum`) | EN 200 + KO 200 | 400 |
| 요약 환각 | HaluEval summarization (EN) + AI-Hub 추상 요약 사실성 검증 (KO, [수동 준비](#ai-hub-한국어-사실성-데이터-수동-준비)) | EN 400 + KO 400 | 800 |
| 분류 | MASSIVE intents (`mteb/amazon_massive_intent`) | EN 1,000 + KO 1,000 | 100 (20개씩 배치) |

**요약**은 ROUGE-Lsum, chrF++, multilingual BERTScore F1, compression ratio, 출력 길이,
empty output rate, refusal rate로 채점합니다. EN과 KO는 항상 따로 보고하며 절대 평균 내지 않습니다.

**요약 환각**은 `SUPPORTED` / `HALLUCINATED` 이진 판정이며 accuracy, macro-F1,
hallucination precision/recall/F1, supported recall, invalid output rate를 냅니다.

**분류**는 MASSIVE semantic id를 공유하는 EN/KO 쌍에 대한 60-intent 분류이며 accuracy,
macro-F1, 그리고 cross-lingual consistency(both correct / EN only / KO only / both wrong)를 냅니다.

여기에 생성된 요약에 대한 **surface fact support 진단 지표**가 추가됩니다. 무엇이고 무엇이
아닌지는 [결과 읽는 법](#결과-읽는-법)을 참고하세요.

위 숫자 중 어느 것도 LLM-as-a-Judge를 쓰지 않습니다. Judge는 제공되지만 기본 비활성이고
보조 지표입니다. [LLM-as-a-Judge (선택)](#llm-as-a-judge-선택) 참고.

---

## 준비

### 1. 설치

Python 3.12 이상과 [uv](https://docs.astral.sh/uv/)가 필요합니다.

```bash
uv sync
```

선택적 extras:

```bash
uv sync --extra bertscore   # BERTScore 활성화 (torch + transformers, 수 GB)
uv sync --extra dev         # 테스트 의존성
```

`bertscore` extra 없이도 벤치마크는 끝까지 동작합니다. 해당 지표는 오해를 부르는 `0.0` 이 아니라
사유와 함께 `null` 로 보고됩니다.

### 2. OpenRouter 계정과 API 키

1. <https://openrouter.ai> 에서 계정을 만들고 크레딧을 충전한 뒤 *Keys* 에서 키를 발급합니다.
2. `cp .env.example .env` 후 `OPENROUTER_API_KEY` 를 채웁니다.

`OPENROUTER_HTTP_REFERER` 와 `OPENROUTER_APP_NAME` 은 선택적인 attribution 메타데이터입니다.

**이 프로젝트에는 vendor API 키가 어디에도 없습니다.** `OPENAI_API_KEY`도, `GOOGLE_API_KEY`도,
`MISTRAL_API_KEY`도, BYOK vendor key도 없습니다. 모델 API와 통신하는 것은 provider adapter
하나(`OpenRouterClient`)뿐이며, vendor 전용 헤더가 절대 전송되지 않는다는 것을 테스트가 검증합니다.

### 3. 데이터 준비

```bash
benchmark prepare
```

공개 데이터셋을 커밋 SHA에 고정해 한 번만 내려받고(약 320 MB), 샘플을 결정적으로 선택해
`data/manifests/` 에 frozen manifest를 씁니다. 몇 분 걸리며, 명시적으로 요청하지 않는 한 다시
실행되지 않습니다.

---

## 지원 모델

모든 id는 OpenRouter model slug입니다. `benchmark models` 로 최신 표를 볼 수 있습니다.

| Slug | 이름 | Input $/M | Output $/M |
|---|---|---:|---:|
| `openai/gpt-5.6-luna` | GPT-5.6 Luna | 0.20 | 1.20 |
| `google/gemini-3.1-flash-lite` | Gemini 3.1 Flash-Lite | 0.25 | 1.50 |
| `google/gemini-2.5-flash-lite` | Gemini 2.5 Flash-Lite | 0.10 | 0.40 |
| `mistralai/mistral-small-2603` | Mistral Small 4 | 0.15 | 0.60 |
| `mistralai/ministral-14b-2512` | Ministral 3 14B | 0.20 | 0.20 |
| `mistralai/ministral-8b-2512` | Ministral 3 8B | 0.15 | 0.15 |
| `mistralai/ministral-3b-2512` | Ministral 3 3B | 0.10 | 0.10 |

기본 벤치마크는 **input $0.25/M, output $1.50/M 이하** 모델을 대상으로 합니다. 범위를 벗어나면
`benchmark models` 가 표시해 줍니다.

### 새 모델 추가하기

코드 수정 없이 YAML 두 곳만 고치면 됩니다.

```yaml
# config/models.yaml
models:
  - model_id: vendor/new-cheap-model-2026
    display_name: New Cheap Model
    vendor: vendor
    enabled: true
    max_output_tokens: 256
    reasoning:
      enabled: false
    routing:
      mode: default
      provider_order: null
      allow_fallbacks: true
```

```yaml
# config/pricing.yaml
models:
  vendor/new-cheap-model-2026:
    input_per_million: 0.12
    output_per_million: 0.50
    cached_input_per_million: null
```

그다음:

```bash
benchmark estimate --model vendor/new-cheap-model-2026
benchmark run      --model vendor/new-cheap-model-2026
benchmark report
```

모든 항목의 `provider` 는 반드시 `openrouter` 여야 하며, 다른 값은 로더가 거부합니다.
vendor별 차이(thinking 모드, 출력 상한, 라우팅)는 모델 항목의 설정으로 표현하고,
벤치마크 코드에서 분기하지 않습니다.

### 가격 갱신

```bash
benchmark pricing              # OpenRouter 실시간 가격을 조회하고 config와 비교
benchmark pricing --no-refresh # 오프라인으로 로컬 저장값만 확인
```

pricing snapshot을 DB에 저장하고 `config/pricing.yaml` 대비 모델별 차이를 출력합니다.
가격이 바뀌면 YAML을 갱신하세요. 이 값은 예산 추정 전용이므로 **가격을 바꿔도 캐시된 추론은
단 하나도 무효화되지 않습니다.**

### 모델 파라미터 호환성

OpenRouter는 모델마다 받아들이는 요청 파라미터가 다릅니다. 예를 들어 `openai/gpt-5.6-luna`
같은 추론 모델은 `temperature` 와 `top_p` 를 지원하지 않고, `mistralai/ministral-8b-2512` 는
`reasoning` 을 지원하지 않습니다.

기본 라우팅은 `require_parameters: true` 입니다. 이 설정은 endpoint가 `temperature=0` 을 조용히
무시하는 것을 막아 주지만, 대신 **모델이 지원하지 않는 파라미터를 보내면 모든 endpoint가 걸러져서
HTTP 404 `No endpoints found that can handle the requested parameters` 가 돌아옵니다.**

그래서 각 모델에 실제 지원 여부를 기록하고, 지원하지 않는 파라미터는 요청에서 아예 뺍니다.

```yaml
- model_id: openai/gpt-5.6-luna
  parameters:
    temperature: false   # OpenRouter가 temperature 미지원으로 보고
    top_p: false
    stop: false
```

빠진 파라미터는 cache key에서도 빠집니다. cache key는 "실제로 무엇을 요청했는가" 를 기술해야
하기 때문입니다.

설정이 OpenRouter의 현재 메타데이터와 맞는지 확인하려면:

```bash
benchmark models --check
```

불일치가 있으면 어떤 모델의 어떤 플래그를 어떻게 고쳐야 하는지 출력하고 exit code 1로 끝납니다.
`benchmark run` 도 시작 전에 같은 검사를 수행하며(이미 가격 조회로 받아온 `/models` 응답을
재사용하므로 추가 비용 없음), 문제가 있으면 **요청을 하나도 보내지 않고** 중단합니다.
검사를 건너뛰려면 `--skip-preflight` 를 쓰세요.

### Routing 설정

OpenRouter는 같은 slug를 여러 upstream endpoint로 라우팅할 수 있으므로, 라우팅 설정도
재현성 정보로 취급합니다.

```yaml
routing:
  mode: pinned                  # default | pinned | restricted
  provider_order: [deepinfra]   # mode가 "pinned"이면 필수
  allow_fallbacks: false
  only: null
  ignore: [some-provider]
  require_parameters: true      # temperature/max_tokens를 무시하는 endpoint 제외
  data_collection: allow
  sort: null                    # price | throughput | latency
```

이 값들은 OpenRouter 요청의 `provider` 필드로 전달되며, upstream provider의 API를 직접
호출하는 일은 없습니다. 이 블록의 해시는 inference cache key의 일부이므로 **라우팅을 바꾸면
해당 캐시 항목이 무효화됩니다.** OpenRouter가 알려주는 경우 각 요청을 실제로 처리한
resolved model과 upstream provider도 결과와 함께 저장합니다.

---

## 데이터 준비 상세

`benchmark prepare` 는 manifest 네 개와 judge 부분집합을 만듭니다.

```
data/manifests/
  xlsum_v1.jsonl                       400 cases  (EN 200 / KO 200)
  halueval_summary_v1.jsonl            400 cases  (문서 200 × 2)
  aihub_summary_factuality_v1.jsonl    400 cases  (문서 200 × 2)  -- 선택
  massive_v1.jsonl                   2,000 cases  (EN 1,000 / KO 1,000)
  judge_xlsum_v1.jsonl                 200 cases  (EN 100 / KO 100)
```

각 manifest에는 `.meta.json` 사이드카가 붙어 데이터셋 리비전, 샘플링 시드, 전처리 정책,
언어별 개수, 그리고 전체 레코드에 대한 **manifest hash** 를 기록합니다. manifest를 로드할 때
이 해시를 다시 검증하므로, 몰래 수정된 manifest는 조용히 결과를 바꾸는 대신 즉시 에러를 냅니다.

manifest는 다시 샘플링되지 않습니다. 재생성은 명시적으로만 가능합니다.

```bash
benchmark prepare --force
```

canonical 전처리(NFC 정규화, 공백 정리, 요약용 약 1,800토큰 / 환각용 약 1,600토큰 절단)는
모델이 아니라 **prepare 단계에서 한 번** 적용됩니다. 따라서 truncation은 frozen sample의 속성이며,
모델마다 다르게 적용되는 일이 없습니다.

### AI-Hub 한국어 사실성 데이터 (수동 준비)

AI-Hub *추상 요약 사실성 검증* 데이터(AI-Hub 157)는 라이선스가 걸려 있습니다. 이 프로젝트는
**절대 자동으로 내려받거나 스크래핑하지 않습니다.** 직접 준비하세요.

1. <https://aihub.or.kr> 에 로그인해 데이터 이용을 신청하고 내려받습니다.
2. 아무 로컬 경로에나 둡니다. **압축을 풀 필요 없습니다.** 배포된 `.zip` 을 그대로 읽습니다.
3. 환경변수로 경로를 지정하고 `prepare` 를 다시 실행합니다.

```bash
export AIHUB_FACTUALITY_DATA_PATH=/path/to/157-추상요약사실성검증데이터
benchmark prepare
```

기대하는 디렉터리 구조는 AI-Hub가 배포하는 그대로입니다.

```
157-추상요약사실성검증데이터/01-1.정식개방데이터/
  Training/01.원천데이터/TS_*.zip          (불필요 - 건너뜀)
  Training/02.라벨링데이터/TL_*_corrected_type*.zip
  Validation/01.원천데이터/VS_*.zip        (불필요 - 건너뜀)
  Validation/02.라벨링데이터/VL_*_corrected_type*.zip
```

라벨링 JSON 하나가 그 자체로 완결적입니다. `original_text` 가 원문, `annotation.original_summary`
가 오류를 포함한 요약, `annotation.corrected_summary.corrected_typeN` 이 주석자의 수정본입니다.
여기서 문서당 두 케이스가 나옵니다 — 오류 요약(`HALLUCINATED`)과 그 수정본(`SUPPORTED`).

선택 기준은 `config/benchmark.yaml` 의 `dataset_sources.aihub` 에서 제어합니다.

```yaml
aihub:
  split: Validation                     # Training | Validation | null (둘 다)
  preferred_error_type: type5
  allow_non_factual_fallback: false
  exclude_omission_only_errors: true
  exclude_broken_prefix_summaries: true
  require_untruncated_source: true
```

**왜 이런 필터가 필요한가.** AI-Hub는 여섯 가지 오류 유형을 주석하는데, 그중 사실 관계 오류는
하나뿐입니다.

| 유형 | 의미 | 사용 |
|---|---|---|
| 1 | 한글 맞춤법, 띄어쓰기 오류 | 아니오 |
| 2 | 단어 선택 오류 | 아니오 |
| 3 | 비문 | 아니오 |
| 4 | 미완성 또는 불완전한 문장 | 아니오 |
| **5** | **키워드 또는 중요 내용 오류** | **예** |
| 6 | 유사한 내용 반복 | 아니오 |

맞춤법을 고쳤다고 해서 요약이 원문에 부합하지 않게 되는 것은 아닙니다. 따라서 1~4, 6번 유형은
환각으로 잘못 라벨링하는 대신 제외합니다. type5 안에서도 두 가지를 더 걸러냅니다.

* **omission-only 문서 제외.** type5는 *틀린 내용* 과 *빠진 내용* 을 같이 묶습니다. 주석된 모든
  구간이 순수 삽입인 경우 — 즉 수정본이 빠뜨린 절을 끼워넣기만 한 경우 — "오류" 요약이 하는
  주장은 여전히 전부 원문에 의해 뒷받침되므로 `HALLUCINATED` 라고 부르면 안 됩니다. 문자 단위
  diff로 판정합니다(부분문자열 검사로는 안 됩니다. 빠진 절은 대개 구간 *중간* 에 삽입됩니다).
  Validation split 기준 type5의 약 28%가 여기 해당합니다.
* **첫 음절이 잘린 요약 제외.** 일부 레코드는 단어 중간에서 시작합니다("트럼프 대통령" 대신
  "럼프 대통령"). 이는 사실성이 아니라 OCR 내성을 시험하는 셈입니다. 기계요약의 약 0.5%,
  소수인 사람요약의 약 48%가 해당됩니다.

마지막으로, 원문이 1,600토큰 절단을 견디지 못하는 문서는 제외합니다. 근거 문장이 잘려나가면
`SUPPORTED` 라벨을 검증할 수 없기 때문입니다.

Validation split에서 이 규칙들을 적용하면 약 1,900개 문서가 남고, 그중 200개를 결정적으로
선택합니다. 각 필터가 몇 개를 제외했는지는 manifest의 `.meta.json` 에 기록됩니다.

**배포 구조와 다른 형태로 갖고 계시다면**, 아래 키를 가진 JSONL도 읽을 수 있습니다.

```json
{"id": "doc-001", "source": "원문 …", "erroneous_summary": "오류가 있는 요약 …", "corrected_summary": "수정된 요약 …", "error_type": "type5"}
```

**이 데이터가 없으면 한국어 환각 벤치마크는 `SKIPPED` 로 표시되고 나머지는 정상 실행됩니다.**
한국어 환각 점수를 임의로 지어내지 않습니다.

---

## 벤치마크 실행

```bash
benchmark run --model openai/gpt-5.6-luna
benchmark run --all
benchmark run --model google/gemini-2.5-flash-lite --benchmark summarization
benchmark run --model mistralai/ministral-8b-2512 --language ko
```

주요 플래그:

| 플래그 | 효과 |
|---|---|
| `--dry-run` | 계획·가격·예산만 확인. **추론 API를 절대 호출하지 않음.** |
| `--budget-usd 1.00` | 모델당 예산 변경 |
| `--allow-over-budget` | 추정 비용이 예산을 넘어도 실행 |
| `--retry-failed` | 실패로 저장된 케이스도 다시 시도 |
| `--force` | 성공 캐시를 무시하고 전부 다시 실행 |
| `--no-metric-cache` | 캐시된 추론으로 지표만 다시 계산 |
| `--judge cheap\|strong` | 선택적 judge 단계 추가 |
| `--skip-preflight` | 모델이 거부하는 파라미터가 있어도 강행 |
| `-v` / `-vv` | 콘솔 로그를 INFO / DEBUG 로 |
| `--log-file PATH` | 로그 파일 위치 변경 |

### Dry run

```bash
benchmark run --model google/gemini-3.1-flash-lite --dry-run
```

벤치마크·언어별 pending/cached 분포, 예상 신규 토큰, 예상 신규 비용, 예산,
`SAFE TO RUN` / `OVER BUDGET` 판정을 출력합니다. dry run에서 pricing 메타데이터 조회는
허용되지만 생성 요청은 불가능합니다. 시도하면 클라이언트가 `DryRunViolation` 을 던집니다.

---

## 로깅과 실패 중단

### 로그

모든 실행은 `data/logs/benchmark-<timestamp>.log` 에 DEBUG 레벨 로그를 남깁니다. 플래그가
필요 없습니다. 요청이 실패하면 모델, 샘플 id, HTTP 상태, OpenRouter 에러 메시지, 시도 횟수가
전부 기록됩니다.

콘솔은 기본적으로 조용합니다(WARNING 이상). 진행 상황이 궁금하면 `-v`(INFO), 요청 단위까지
보고 싶으면 `-vv`(DEBUG)를 쓰세요.

```bash
benchmark run --model google/gemini-2.5-flash-lite -v
benchmark run --model google/gemini-2.5-flash-lite --log-file /tmp/run.log
```

오래된 로그는 `logging.keep_last`(기본 50개)만 남기고 자동 정리됩니다.

### 실패 중단 (circuit breaker)

설정 오류는 모든 케이스에서 똑같이 실패합니다. 그대로 두면 1,300건을 전부 요청해서 같은 에러
1,300개를 만들어낼 뿐입니다. 그래서 다음 조건 중 하나라도 걸리면 실행을 중단합니다.

```yaml
failure_guard:
  enabled: true
  max_consecutive_failures: 10      # 연속 실패
  max_failure_rate: 0.5             # 누적 실패율
  min_attempts_before_rate_check: 20
```

중단되면 실행 상태는 `ABORTED` 가 되고, 콘솔에 중단 사유와 **가장 많이 나온 에러 메시지**가
개수와 함께 출력됩니다. 실패 개수만 보여주고 원인을 감추지 않습니다.

중요한 점은 **시도조차 하지 못한 케이스는 아무것도 저장하지 않는다** 는 것입니다. 원인을 고친 뒤
그냥 다시 실행하면 됩니다. `--retry-failed` 도 필요 없습니다. (예산 가드로 중단된 경우도 같습니다.)

## 캐시 동작

`data/cache/benchmark.db`(SQLite, WAL) 안에 서로 독립적인 캐시 두 개가 있습니다.

### Inference cache

각 요청 전에, 모델에게 무엇을 물었는지를 결정하는 모든 정보를 canonical JSON으로 만들어
SHA-256 해시를 계산합니다.

```
api_provider (= openrouter)      benchmark name + version
OpenRouter model slug            dataset revision + split
alias resolution (있는 경우)      sample id + sample content hash
routing config hash              system / user prompt hash
                                 prompt template version
temperature, top_p, seed         reasoning 설정
max output tokens                structured-output schema version
tools / web search 플래그         truncation policy version
                                 generation config version
```

해당 키에 `SUCCESS` 행이 있으면 **OpenRouter를 호출하지 않습니다.** 없으면 요청을 보내고
raw response, usage, latency, resolved model, upstream provider, 실제 비용을 저장합니다.

### Metric cache

독립적으로 키를 만듭니다.

```
inference result hash + evaluator name + evaluator version
+ evaluator config hash + reference hash
```

따라서 BERTScore 체크포인트를 바꾸면 지표만 다시 계산되고 추론은 전부 재사용됩니다.

### 캐시 무효화 규칙

| 변경 | Inference cache | Metric cache |
|---|---|---|
| 가격 변경 (`config/pricing.yaml`, OpenRouter 실시간 가격) | **유지** | 유지 |
| 리포트 형식 / 리더보드 컬럼 | **유지** | 유지 |
| metric 설정 또는 evaluator 버전 | **유지** | 무효화 |
| 프롬프트 텍스트 또는 prompt template version | 무효화 | (재채점) |
| model slug | 무효화 | (재채점) |
| routing 설정 | 무효화 | (재채점) |
| 생성 파라미터 (temperature, max tokens, reasoning, seed) | 무효화 | (재채점) |
| dataset revision, 샘플 내용, truncation policy | 무효화 | (재채점) |
| `benchmark_version` / `generation.version` 상향 | 무효화 | (재채점) |

첫 줄이 핵심입니다. **가격이 바뀌었다는 이유로 API를 다시 호출하는 일은 없어야 하며**,
이를 단위 테스트가 강제합니다.

### 재시도 정책

* `SUCCESS` 결과는 다시 요청하지 않습니다. `--force` 로만 무시할 수 있습니다.
* 저장된 실패는 조용히 재시도하지 않습니다. `--retry-failed` 를 명시해야 합니다.
* 모든 물리적 HTTP 시도는 실패 포함 `inference_attempts` 에 저장됩니다. 실패한 시도에 과금이
  발생했다면 그 `usage.cost` 도 보존되어 합산됩니다.
* 재시도는 jitter가 섞인 exponential backoff이며 `Retry-After` 를 존중합니다.
  `config/benchmark.yaml` 의 `retry:` 에서 설정합니다.

상태값: `SUCCESS`, `OPENROUTER_ERROR`, `RATE_LIMIT`, `PARSE_ERROR`, `INVALID_OUTPUT`,
`BUDGET_STOPPED`, `SKIPPED`.

응답은 왔는데 레이블로 파싱할 수 없는 경우는 전송 실패가 아니라 `SUCCESS` 로 저장하고
`invalid_output_rate` 에 반영합니다. 요청 자체는 성공했고, temperature 0 모델에 다시 물어도
똑같이 파싱 불가능한 텍스트가 나오기 때문입니다. `PARSE_ERROR` 와 `INVALID_OUTPUT` 은
비정상 HTTP 응답과 JSON이 아닌 judge 응답에만 사용합니다.

캐시는 언제든 확인할 수 있습니다.

```bash
benchmark cache stats
benchmark cache clear-failed --model google/gemini-2.5-flash-lite
```

---

## 비용

### `usage.cost` 가 정답이다

OpenRouter는 모든 응답에 `usage` 블록을 반환합니다(클라이언트가 항상 `"usage": {"include": true}`
를 보냅니다). 그 안의 `cost` 가 해당 요청에 실제로 청구된 금액이며, 이 프로젝트가 보고하는 값입니다.

```
usage.prompt_tokens                     usage.cost           <- 실제 청구액
usage.completion_tokens                 usage.cost_details
usage.total_tokens
usage.prompt_tokens_details.cached_tokens
usage.prompt_tokens_details.cache_write_tokens
usage.completion_tokens_details.reasoning_tokens
```

가격표 계산은 두 가지 용도로만 씁니다. 실행 전 예산 추정, 그리고 실행 후 청구액 검증입니다.
둘이 다르면 리포트는 `usage.cost` 를 보여주고 차이를 함께 표시합니다.

### 비용 가드

모델당 기본 예산은 **$1.00** 입니다(`budget.default_usd`). 요청을 보내기 전에 *pending* 작업의
예상 비용(캐시 히트 제외)을 계산하고, 예산을 넘으면 실행을 거부합니다.

```bash
benchmark run --model google/gemini-3.1-flash-lite --budget-usd 2.00
benchmark run --model google/gemini-3.1-flash-lite --allow-over-budget
```

실행 중에도 누적 `usage.cost` 를 추적하다가 예산을 넘기기 전에 새 요청을 중단합니다. 이때까지의
결과는 저장되고 실행은 `BUDGET_STOPPED` 로 끝납니다. Judge 비용은 벤치마크 예산과 **분리해서**
추적·표시합니다.

### 전체 실행 비용

전체 벤치마크의 보수적 토큰 envelope:

| | Input | Output |
|---|---:|---:|
| 요약 | 720K | 64K |
| 환각 | 1.28M | 6.4K |
| 분류 | ~185K | ~10K |
| **합계** | **≈2.185M** | **≈80.4K** |

가격 상한($0.25/M in, $1.50/M out) 기준 **≈$0.667** 입니다. 설정된 가격 기준 모델별 예상:

| 모델 | 예상 |
|---|---:|
| `openai/gpt-5.6-luna` | ≈$0.5335 |
| `google/gemini-3.1-flash-lite` | ≈$0.6669 |
| `google/gemini-2.5-flash-lite` | ≈$0.2507 |
| `mistralai/mistral-small-2603` | ≈$0.3760 |
| `mistralai/ministral-14b-2512` | ≈$0.4531 |
| `mistralai/ministral-8b-2512` | ≈$0.3398 |
| `mistralai/ministral-3b-2512` | ≈$0.2265 |

어디까지나 추정치입니다. 최종 금액은 언제나 `usage.cost` 의 합입니다.

### 비용 블록 읽는 법

```
Fresh cost this run:       $0.0000   <- 이번 실행에서 청구된 금액
Stored benchmark cost:     $0.2487   <- 이 모델 캐시를 만드는 데 지금까지 지불한 총액
Estimated cost w/o cache:  $0.2487   <- 캐시가 없었다면 이번 실행에 들었을 비용
Estimated cache savings:   $0.2487   <- 그 차액
```

이미 끝난 벤치마크를 다시 실행하면 **`Fresh cost this run: $0.0000`** 이 나오며, 동시에
그 캐시를 만드는 데 원래 얼마가 들었는지도 함께 보고합니다.

---

## LLM-as-a-Judge (선택)

기본 비활성입니다(`config/judge.yaml`). Judge 역시 다른 모든 것과 마찬가지로 OpenRouter를
통해서만 호출합니다.

```bash
benchmark run --model google/gemini-3.1-flash-lite --judge none
benchmark run --model google/gemini-3.1-flash-lite --judge cheap
benchmark run --model google/gemini-3.1-flash-lite --judge strong --judge-samples 200
```

* `cheap` → `openai/gpt-5.6-luna`
* `strong` → `openai/gpt-5.6-sol` (**가격을 코드에 박지 않았습니다.** 실행 시점에 OpenRouter에서
  조회합니다)

Judge는 XL-Sum manifest에서 고정된 200 케이스(EN 100 / KO 100)를 대상으로 factual consistency,
key information coverage, conciseness, overall quality를 케이스당 structured JSON 하나로 채점합니다.
Judge 결과도 추론과 같은 규율로 캐시됩니다. 요약·judge 모델·프롬프트·rubric·생성 설정·라우팅이
같으면 다시 호출하지 않습니다.

Judge와 평가 대상이 같은 slug이면 리포트에 `SELF_JUDGE_WARNING` 을 표시하고 해당 점수를
참고용으로 표시합니다. Judge 출력은 절대 primary factuality score가 되지 않습니다.

---

## 결과 읽는 법

```bash
benchmark report
```

다음을 생성합니다.

```
results/
  leaderboard.csv
  leaderboard.json
  leaderboard.md
  report.html
  models/{safe_model_slug}/summary.json     예: google__gemini-3.1-flash-lite/
```

raw OpenRouter 응답은 DB에만 저장하고 리포트에는 싣지 않습니다.

리더보드 컬럼은 모델, input/output 가격, 요약 EN/KO, 환각 F1 EN/KO, 분류 EN/KO,
EN-KO consistency, 토큰 합계, OpenRouter 비용, judge 비용입니다. 모델 상세에는 ROUGE-Lsum,
BERTScore, chrF++, surface fact support, compression ratio, 환각 accuracy/macro-F1/recall,
분류 accuracy/macro-F1, cross-lingual 분해가 추가됩니다.

### 숫자를 인용하기 전에 알아둘 것

**성격이 다른 지표를 하나의 composite score로 합치지 않습니다.** 요약 ROUGE와 intent 분류
accuracy는 같은 축이 아니며, 평균을 내면 이 벤치마크가 드러내려는 트레이드오프가 가려집니다.

**Surface fact support는 진단 지표이지 환각 지표가 아닙니다.** 생성된 요약에서 숫자, 백분율,
날짜, 통화 금액, 정밀도 높은 개체명 후보를 뽑아 정규화 후 원문과 대조합니다.

```
surface_fact_support_precision = 뒷받침되는 추출 사실 / 전체 추출 사실
```

명세가 든 예시(원문 12%, 요약 21%)를 잡아내며 `number_mismatch`, `percentage_mismatch`,
`date_mismatch`, `currency_mismatch`, `unsupported_named_entity` 를 따로 집계합니다. 다만
의미 수준의 불충실("부인했다" vs "확인했다")은 보지 못하고, 누락도 보지 못하며, 개체명 후보는
의도적으로 고정밀·저재현으로 뽑습니다. 점수가 깨끗하다고 해서 충실한 요약이라는 뜻은 아닙니다.

**한국어 ROUGE는 문자 단위 토큰화를 씁니다.** `rouge-score` 의 기본 토크나이저는 `[a-z0-9]`
이외 문자를 모두 제거해서 한국어 텍스트를 통째로 날려버리고 모든 모델에 0.0을 줍니다. 그래서
다국어 토크나이저를 넣었습니다. 라틴 문자는 단어 단위, 한글/CJK는 문자 단위입니다. 따라서
한국어 ROUGE는 *이 벤치마크 안의 모델 간 비교* 에는 유효하지만, 단어 단위인 영어 ROUGE와
직접 비교할 수는 없습니다.

**EN과 KO 환각 점수는 엄밀히 비교 대상이 아닙니다.** 영어 케이스는 HaluEval로, LLM이 올바른
요약을 일부러 망가뜨린 것입니다. 한국어 케이스는 AI-Hub로, 사람이 실제 기계 생성 요약을
교정한 것입니다. 따라서 한국어 `HALLUCINATED` 후보는 실제 모델 출력이라 불충실할 뿐 아니라
비문인 경우가 많고, 영어 쪽은 구조상 유창합니다. 두 열을 가로질러 비교하지 말고, 각 언어
열 안에서 모델끼리 비교하세요.

**Invalid output은 오답으로 셉니다.** 파싱 불가능한 응답을 낸 모델이 해당 케이스를 분모에서
빼는 이득을 얻지 않습니다. 점수가 낮은 이유를 볼 수 있도록 `invalid_output_rate` 를 함께 보고합니다.

**Cross-lingual consistency** 는 같은 semantic id를 공유하는 EN/KO 쌍에서 *정오답이 일치하는지*
를 봅니다. 정답 여부와 무관하게 *같은 intent를 예측했는지* 는 `prediction_agreement` 로 따로 냅니다.

**BERTScore는 `null` 일 수 있습니다.** 선택적 extra가 설치되지 않은 경우이며, 사유와 함께
표시되고 절대 `0.0` 으로 처리하지 않습니다.

---

## 문제 해결

### `No endpoints found that can handle the requested parameters` (HTTP 404)

모델이 지원하지 않는 요청 파라미터를 보냈다는 뜻입니다.

```bash
benchmark models --check
```

가 어떤 모델의 어떤 플래그를 고쳐야 하는지 알려줍니다. `config/models.yaml` 의 해당 모델에
`parameters:` 블록을 추가하고 다시 실행하세요. 자세한 내용은
[모델 파라미터 호환성](#모델-파라미터-호환성) 참고.

### 실행이 `ABORTED` 로 끝났다

circuit breaker가 동작한 것입니다. 콘솔의 에러 메시지 표와 `data/logs/` 의 로그 파일을 보세요.
시도되지 않은 케이스는 저장되지 않았으므로, 원인을 고친 뒤 같은 명령을 다시 실행하면 됩니다.

### 실행이 `BLOCKED` 로 끝났다

요청을 하나도 보내지 않고 멈춘 상태입니다. 원인은 둘 중 하나입니다. 예상 비용이 예산을 넘었거나
(`--budget-usd` / `--allow-over-budget`), 파라미터 호환성 검사에 걸렸거나
(`benchmark models --check`).

### 전부 실패했는데 비용이 $0 이다

정상입니다. OpenRouter가 요청을 거부하면 과금되지 않습니다. 반대로 과금된 실패는
`usage.cost` 가 보존되어 `retry_error_cost` 에 합산됩니다.

### 이미 쌓인 실패 결과를 지우고 싶다

```bash
benchmark cache clear-failed --model <slug>
```

성공한 결과는 건드리지 않습니다.

## 재현성

모든 실행은 `summary.json` 과 `runs` 테이블에 다음을 기록합니다. 벤치마크 git commit과 버전,
데이터셋 리비전과 manifest 해시, 프롬프트 버전, OpenRouter model slug, 가능한 경우 resolved
model과 upstream provider, 라우팅 설정, 생성·reasoning 파라미터, metric 버전, 실행 타임스탬프,
사용된 OpenRouter pricing snapshot, 실제 `usage.cost`, Python/패키지 버전.

전 구간에서 고정 model slug를 씁니다. `:latest` 형태의 alias는 권장하지 않으며, 사용하는 경우
마지막으로 관측된 resolution이 cache key에 포함됩니다. 따라서 resolution이 바뀌면 두 모델이
조용히 섞이는 대신 새로운 benchmark revision으로 취급됩니다.

---

## 프로젝트 구조

```
config/          models.yaml, pricing.yaml, benchmark.yaml, judge.yaml
src/llmbench/
  openrouter/    유일한 provider adapter (client, types)
  cache/         inference + metric cache key, metric cache
  db/            SQLAlchemy 스키마와 store facade
  datasets/      리비전 고정 리더, frozen manifest, prepare
  benchmarks/    벤치마크별 task 생성 + 채점
  metrics/       ROUGE, chrF++, BERTScore, surface facts, label metrics
  prompts/       버전이 붙은 EN/KO 프롬프트 템플릿
  judge/         선택적 LLM-as-a-Judge
  reporting/     모델별 summary, 리더보드, HTML
  runner.py      plan -> run -> score -> persist
  budget.py      추정과 지출 가드
  failures.py    실패 중단 (circuit breaker)
  logging_setup.py  콘솔/파일 로깅
  pricing.py     들어올 땐 추정치, 나갈 땐 usage.cost
  cli.py         `benchmark` 커맨드
tests/           단위 테스트 + mock transport 기반 end-to-end 벤치마크
scripts/         전체 규모 mock end-to-end 데모
```

provider adapter는 정확히 하나입니다. `OpenAIProvider`, `GoogleProvider`, `MistralProvider`,
`AnthropicProvider` 같은 클래스는 없으며, 벤치마크 코어가 vendor를 보고 endpoint를 분기하는
일도 없습니다.

---

## 테스트

```bash
uv run pytest -q
```

전체 규모 데모도 있습니다. 실제 벤치마크(frozen 케이스 3,200개, 요청 1,300건)를 같은 mock
transport로 실행합니다.

```bash
uv run python scripts/mock_end_to_end.py
uv run python scripts/mock_end_to_end.py --out results/sample-mock-run
```

두 번째 동일 실행이 추론 요청을 정확히 0건 보내는지, 보고된 비용이 mock의 `usage.cost` 합과
일치하는지, cache savings가 캐시 덕분에 다시 지불하지 않은 금액과 같은지를 단언합니다.
출력물은 `results/sample-mock-run/` 에 있으며, 그 숫자는 실제 모델이 아니라 mock의 동작을
나타냅니다. 해당 디렉터리의 `README.txt` 에도 그렇게 적혀 있습니다.

테스트 스위트는 네트워크도 OpenRouter 키도 없이 동작합니다. end-to-end 테스트는 mock provider
클래스가 아니라 **mock HTTP transport** 를 사용하므로 실제 클라이언트, 재시도 루프, 캐시,
예산 가드, 지표, 리포팅이 모두 그대로 실행됩니다. 검증 항목에는 다음이 포함됩니다.
동일 요청의 캐시 히트, 가격 변경이 추론 캐시를 무효화하지 않음, 프롬프트/slug/라우팅 변경 시
캐시 미스, metric 버전 변경 시 추론 재사용, 실패한 시도의 비용 보존, 예산 가드, MASSIVE EN/KO
id 정합성, manifest 안정성, dry run이 추론 호출을 하지 않음, judge 비활성 시 judge 요청 없음,
두 번째 실행의 추론 호출 0건, `usage.cost` 와 보고 비용의 일치, vendor API 키 불필요.

---

## 라이선스

벤치마크 코드는 MIT입니다. 데이터셋은 각자의 라이선스를 따릅니다. XL-Sum(CC BY-NC-SA 4.0),
HaluEval(MIT, CNN/DailyMail 기반), MASSIVE(CC BY 4.0), 그리고 AI-Hub 사실성 데이터는 AI-Hub
약관을 따릅니다. 마지막 데이터를 이 프로젝트가 절대 내려받거나 재배포하지 않는 이유가 그것입니다.
