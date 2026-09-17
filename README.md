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

세 가지 능력을 **영어와 한국어에서 각각** 측정합니다. 모델 하나당 3,200 케이스,
OpenRouter 요청 1,300건입니다.

| 벤치마크 | 데이터셋 | 과제 | 케이스 | 요청 수 |
|---|---|---|---:|---:|
| [요약](#1-요약-summarization) | XL-Sum (`csebuetnlp/xlsum`) | 뉴스 기사를 1~3문장으로 요약 | EN 200 + KO 200 | 400 |
| [요약 환각 탐지](#2-요약-환각-탐지-hallucination-detection) | HaluEval (EN) + AI-Hub 추상 요약 사실성 검증 (KO, [수동 준비](#ai-hub-한국어-사실성-데이터-수동-준비)) | 요약이 원문에 뒷받침되는지 판정 | EN 400 + KO 400 | 800 |
| [분류](#3-분류-classification) | MASSIVE (`mteb/amazon_massive_intent`) | 발화를 60개 인텐트 중 하나로 분류 | EN 1,000 + KO 1,000 | 100 |

세 벤치마크 모두 **frozen manifest** 를 씁니다. 모든 모델이 같은 샘플을, 같은 전처리로,
같은 프롬프트로 받습니다.

아래 지표 표의 방향 표기는 **↑ 높을수록 좋음 · ↓ 낮을수록 좋음 · — 좋고 나쁨이 아니라 해석용**
입니다. 같은 설명이 `RESULT.md` 의 각 표 아래와 스프레드시트의 `Legend` 시트에도 들어갑니다.

### 1. 요약 (Summarization)

BBC 뉴스 기사(XL-Sum)를 1~3문장으로 요약시키고, 기사에 딸린 **사람이 쓴 참조 요약**과 비교합니다.
원문은 약 1,800토큰으로 잘라 모든 모델에 동일하게 제공하고, 출력은 200토큰까지 받습니다.

| 지표 | 방향 | 의미 |
|---|:--:|---|
| ROUGE-Lsum | ↑ | 참조 요약과의 최장 공통 부분열 기반 F1 (0~1). 문장 단위로 계산합니다. |
| chrF++ | ↑ | 문자 n-gram + 단어 bigram F-score (0~100). 어형 변화에 강해 한국어에 비교적 공정합니다. |
| BERTScore F1 | ↑ | 다국어 임베딩 기반 의미 유사도 (0~1). 표현이 달라도 뜻이 같으면 점수를 줍니다. [선택 설치](#1-설치). |
| Surface fact support | ↑ | 아래 [진단 지표](#부가-surface-fact-support-진단) 참고. |
| Compression ratio | — | 출력 길이 ÷ 원문 길이. 낮으면 압축적, 높으면 장황. **낮다고 무조건 좋은 것은 아닙니다.** |
| 출력 길이 | — | 평균 출력 토큰 수(추정). |
| Empty output rate | ↓ | 빈 출력 비율. |
| Refusal rate | ↓ | 거부 응답 비율. |

빈 출력과 거부는 **분모에서 빼지 않습니다.** 5분의 1을 거부하는 모델이 나머지만으로 높은
점수를 받는 일이 없도록 하기 위해서입니다.

요약은 정답이 하나가 아니라서 ROUGE 절대값 자체는 낮게 나옵니다(0.15~0.25 수준이 정상).
**모델 간 상대 비교**로 읽으세요.

### 2. 요약 환각 탐지 (Hallucination detection)

원문과 후보 요약을 함께 주고, 요약의 모든 사실적 주장이 원문에 의해 뒷받침되는지
`SUPPORTED` / `HALLUCINATED` 중 하나로 답하게 합니다. 출력은 8토큰이면 충분합니다.

문서 하나에서 두 케이스가 나옵니다. 충실한 요약(`SUPPORTED`)과 오류가 있는 요약
(`HALLUCINATED`)이라 레이블이 정확히 반반이고, 따라서 **무작위 추측의 정확도는 0.5** 입니다.

| 지표 | 방향 | 의미 |
|---|:--:|---|
| Accuracy | ↑ | 전체 정답률 (0~1). 0.5가 무작위 수준. |
| Macro-F1 | ↑ | 두 레이블 F1의 단순 평균. 한쪽 레이블만 찍는 모델을 걸러냅니다. |
| Hallucination Precision | ↑ | HALLUCINATED 라고 답한 것 중 실제로 맞은 비율. 낮으면 멀쩡한 요약을 환각이라고 의심. |
| Hallucination Recall | ↑ | 실제 HALLUCINATED 중 잡아낸 비율. 낮으면 환각을 놓침. |
| Hallucination F1 | ↑ | 위 둘의 조화평균. 이 벤치마크의 대표값입니다. |
| Supported Recall | ↑ | 실제 SUPPORTED 를 맞힌 비율. **이 값만 낮으면 과도하게 의심하는 모델**입니다. |
| Invalid output rate | ↓ | 레이블로 파싱할 수 없는 출력 비율. 오답으로 계산됩니다. |

Precision 과 Recall 을 같이 보셔야 합니다. 전부 `HALLUCINATED` 라고 답하면 Recall 은 1.0 이지만
Supported Recall 이 0 이 되고 Macro-F1 이 무너집니다.

> **EN 과 KO 는 직접 비교하지 마세요.** 영어(HaluEval)는 LLM 이 올바른 요약을 일부러 망가뜨린
> 것이고, 한국어(AI-Hub)는 사람이 실제 기계 요약을 교정한 것입니다. 한국어 쪽
> `HALLUCINATED` 후보는 실제 모델 출력이라 불충실할 뿐 아니라 비문인 경우가 많습니다.
> 언어별 열 안에서 모델끼리 비교하세요.

### 3. 분류 (Classification)

MASSIVE 음성비서 발화를 60개 인텐트 중 하나로 분류시킵니다. 비용 절감을 위해 **20건씩 묶어**
한 요청으로 보내고 `<발화 번호>: <인텐트 번호>` 형식으로 답하게 합니다.
**무작위 추측의 정확도는 1/60 ≈ 0.017** 입니다.

| 지표 | 방향 | 의미 |
|---|:--:|---|
| Accuracy | ↑ | 정답률 (0~1). |
| Macro-F1 | ↑ | 60개 인텐트별 F1의 평균. 드문 인텐트를 무시하는 모델에 불리하게 작동합니다. |
| Invalid output rate | ↓ | 배치 형식을 못 지켜 답을 읽지 못한 비율. 오답으로 계산됩니다. |

#### 답변 형식: 자유 텍스트 vs 구조화 출력

같은 과제를 두 조건에서 잴 수 있습니다.

```bash
benchmark run --model <slug> --benchmark classification --classification-mode text
benchmark run --model <slug> --benchmark classification --classification-mode json_schema
```

- `text`(기본) — 모델이 `<번호>: <인텐트 번호>` 형식을 직접 지켜야 합니다.
  **분류 능력 + 형식 준수**를 함께 잽니다.
- `json_schema` — OpenRouter structured output 으로 답을 60개 인텐트 집합에 제약합니다.
  형식 실패가 구조적으로 불가능하므로 **분류 능력만** 남습니다.

두 조건은 cache key 가 다르고 결과 파일도 다릅니다(`summary.json` /
`summary.classification-json_schema.json`). 서로 덮어쓰지 않으므로 RESULT.md 의
「분류: 자유 텍스트 vs 구조화 출력」 절에 차이가 표로 나옵니다. 차이가 클수록 그 모델의
text 점수가 분류 실력이 아니라 형식 준수 실패에 발목 잡혀 있었다는 뜻입니다.

어느 쪽이 옳은지는 용도에 달렸습니다. 서비스에 붙일 모델을 고른다면 형식 준수도 능력이고,
순수 분류력을 비교한다면 `json_schema` 쪽이 맞습니다.

#### 로컬 실행 (OpenRouter 없이)

분류 벤치마크는 **이 머신에서 직접 돌리는 모델**도 지원합니다. 설정은
`config/local_models.yaml` 에 따로 있고, `config/models.yaml` 은 OpenRouter 전용으로 남습니다.

```bash
# CPU 휠
uv sync --extra local --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
# 또는 CUDA 휠 (툴킷 버전에 맞는 태그 선택)
uv sync --extra local --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu122

benchmark run --all-local --benchmark classification
```

```yaml
# config/local_models.yaml
models:
  - model_id: local/qwen2.5-1.5b-instruct-q8-gguf
    display_name: Qwen2.5 1.5B Instruct (GGUF Q8_0, local)
    provider: local
    engine: llama_cpp
    source:
      repo: Qwen/Qwen2.5-1.5B-Instruct-GGUF
      filename: qwen2.5-1.5b-instruct-q8_0.gguf
      revision: 91cad51170dc346986eccefdc2dd33a9da36ead9   # 가중치도 고정합니다
    runtime:
      n_ctx: 4096
      n_gpu_layers: 20    # 28개 레이어 중 GPU 에 올릴 수
      n_seq_max: 32       # 분기 브로드캐스트용 KV 시퀀스 수 (PCD 전용)
      temperature: 0.0
```

**GPU 오프로드는 속도 설정이 아닙니다.** CUDA 커널과 CPU 커널은 답이 미세하게 갈립니다 —
같은 발화 240건을 두 방식으로 돌렸더니 **3건(1.2%)이 달랐습니다.** 그래서 `n_gpu_layers` 는
inference cache key 에 포함되고, 값을 바꾸면 벤치마크가 다시 실행됩니다. 두 결과를 한 숫자로
합치면 서로 다른 시스템 둘을 섞는 셈이기 때문입니다.

`n_threads` 와 `n_batch` 는 반대로 처리량만 바꾸므로 cache key 에서 제외됩니다. 스레드 수를
바꿨다고 전체를 다시 돌릴 이유는 없습니다.

참고 수치(GTX 1650 SUPER 4GB, 데스크톱이 2.2GB 점유 → 여유 1.9GB, 발화 2,000건):

| n_gpu_layers | 건당 | 2,000건 |
|---:|---:|---:|
| 0 (CPU) | 1.194 s | 39.8분 |
| 10 | 0.599 s | 20.0분 |
| 20 | 0.395 s | **13.2분** |
| 24 이상 | 가중치가 VRAM 에 안 들어가 프로세스 중단 |

VRAM 이 부족하면 llama.cpp 는 **조용히 CPU 로 내려가지 않고 프로세스를 종료합니다.** 설정과
다른 조건으로 결과가 기록되는 것보다 낫기 때문에 그대로 둡니다. 여유 VRAM 에 맞춰
`n_gpu_layers` 를 낮추세요.

**디코딩 방식 두 가지를 나란히 잽니다.** 가중치는 같고 답을 만드는 방법만 다릅니다.

| engine | 방식 | 설명 |
|---|---|---|
| `llama_cpp_pcd` | Parallel Constrained Decoding | 프롬프트를 한 번 prefill 하고, **로짓을 후보 집합으로 슬라이싱** 해서 답을 읽어냅니다. 생성하지 않습니다. |
| `llama_cpp` | 문법 유도 생성 (baseline) | GBNF 문법으로 마스킹하며 토큰을 한 개씩 생성합니다. |

기본 설정에 들어 있는 로컬 모델입니다.

| model_id | 가중치 | engine |
|---|---|---|
| `local/qwen2.5-1.5b-instruct-q8-gguf-pcd` | Qwen2.5 1.5B Instruct Q8_0 | PCD |
| `local/qwen2.5-1.5b-instruct-q8-gguf` | 같은 가중치 | GBNF (구현 교차 검증용) |
| `local/qwen3.5-2b-q8-gguf` | Qwen3.5 2B Q8_0 (`unsloth`) | PCD |
| `local/qwen3.5-4b-q4km-gguf` | Qwen3.5 4B Q4_K_M (`unsloth`) | PCD |

둘 다 파싱 불가능한 답이 구조적으로 나올 수 없습니다. 호스팅 `json_schema` 가 제공자에게
사서 쓰는 보장을 로컬에서 직접 수행하는 것입니다.

#### Parallel Constrained Decoding

기법 출처는 <https://huggingface.co/harshatheg/Qwen-2.5-1B-RLCD> 입니다. 동작은 이렇습니다.

1. **단일 prefill** — 문맥을 한 번만 평가해 KV 캐시에 담습니다. 발화마다 시스템 메시지(인텐트
   60개 목록)가 동일하므로 그 접두부는 재사용됩니다.
2. **서브-보캡 로짓 슬라이싱** — 그 위치의 로짓에서 후보 토큰만 뽑아 점수를 매깁니다.
   생성이 없으므로 답이 몇 토큰이든 forward 한 번으로 후보 전체를 평가합니다.
3. **KV 캐시 브로드캐스트 + 토큰 트리** — 접두부를 공유하는 후보는 분기마다 **자기 KV 시퀀스**를
   받고(`kv_cache_seq_cp` 로 prefill 셀을 공유), **트리 한 깊이를 배치 디코딩 한 번**으로
   평가합니다. 분기 수만큼 forward 하지 않습니다.
4. **캘리브레이션된 확률** — 후보 슬라이스에 대한 softmax 로 확신도가 함께 나옵니다.

브로드캐스트를 하려면 컨텍스트가 여러 시퀀스를 지원해야 합니다. llama.cpp 기본값은
`n_seq_max = 1` 이고 `llama_cpp.Llama` 는 이 값을 생성자 인자로 노출하지 않으므로, 이 엔진은
`Llama` 래퍼 대신 `LlamaModel` + `LlamaContext` 를 직접 만들어 `n_seq_max` 를 지정합니다
(라이브러리 상한은 256). `runtime.n_seq_max` 로 조정합니다.

동시에 살아 있는 분기가 `n_seq_max` 를 넘으면 **오류로 중단합니다.** 잘라내면 탐색이 조용히
바뀌어 argmax 가 달라질 수 있기 때문입니다. 시퀀스는 싸니 값을 올리세요.

**프롬프트는 GGUF 에 내장된 chat template 을 씁니다.** ChatML 을 하드코딩하면 Qwen2.5 에서는
맞지만 다른 모델에서는 조용히 틀린 프롬프트가 됩니다. 예를 들어 Qwen3.5 는 thinking 모델이라
assistant 차례에 `<think>\n\n</think>\n\n` 이 붙습니다(비-thinking 모드). 이 자리를 맞추지
않으면 모델이 `<think>` 를 열려는 위치에서 라벨을 채점하게 됩니다.

**원 구현과 두 가지가 다릅니다.**

- 원 구현은 후보를 **첫 토큰만으로** 비교합니다. MASSIVE 60개 인텐트에서는 치명적입니다 —
  `alarm_set` / `alarm_query` / `alarm_remove` 가 모두 토큰 56780 으로 시작해서
  **60개 중 42개가 첫 토큰이 겹칩니다.** 중복된 토큰 id 에 argmax 를 걸면 먼저 온 것이
  조용히 선택됩니다. 원 구현의 Linux/PyTorch 백엔드에는 분기 해소 단계가 아예 없고,
  MLX 백엔드는 자유 생성 후 문자열 매칭으로 때웁니다.
- 여기서는 트리를 **후보가 유일해질 때까지** 내려가며 로그 확률을 누적해 **완전한 시퀀스**로
  비교합니다. 덕분에 greedy 가 아닙니다 — 첫 토큰이 argmax 가 아닌 후보도 총 확률로 이길 수
  있고, 문법 유도 생성은 원리상 그렇게 할 수 없습니다.

**가지치기로 탐색량을 줄입니다.** 로그 확률은 음수라 자식 점수가 부모를 넘을 수 없으므로,
이미 확정된 후보보다 낮은 분기는 탐색하지 않아도 **argmax 가 정확히 보존**됩니다.

실측에서 발화당 **디코딩 2.71회로 분기 7.31개**를 평가합니다(그중 1회는 prefill). 브로드캐스트
없이 분기를 순차로 돌면 같은 작업에 forward 가 그만큼 더 듭니다.

#### 실측 (CPU, 발화 120건, 60개 인텐트)

| 방식 | 건당 | 정확도 | 비고 |
|---|---:|---:|---|
| PCD (KV 브로드캐스트) | **0.569 s** | 0.525 | decode 2.71회, 확신도 제공 |
| PCD (분기 순차 평가) | 0.708 s | 0.525 | 같은 알고리즘, 배치만 없음 |
| 문법 유도 생성 (baseline) | 1.100 s | 0.517 | |

두 방식의 예측은 120건 중 118건이 같습니다. 정확도 차이는 잡음 수준이고, 의미 있는 차이는
**속도(1.55배)와 캘리브레이션된 확신도** 입니다.

원 카드가 내세우는 5.6~7.0배는 **필드가 여러 개인 스키마**(4~28개)에서 나옵니다. 긴 JSON 을
순차 생성하는 대신 모든 필드를 한 번에 평가하기 때문입니다. 이 벤치마크는 필드가 하나라
그 부분의 이득은 적용되지 않고, 남는 이득은 "생성 대신 로짓 읽기" 입니다.

**읽을 때 주의할 점이 두 가지 있습니다.**

- **비용이 없습니다.** 로컬 실행은 과금이 없으므로 비용 열이 `0` 이 아니라 **비어 있습니다**.
  0으로 적으면 리더보드에서 "가장 싼 호스팅 모델"처럼 보이기 때문입니다. `API` 열이 `local`
  인 행이 여기 해당합니다.
- **조건이 다릅니다.** 호스팅 실행은 비용을 아끼려고 발화 20건을 한 요청에 묶지만, 로컬은
  아낄 API 호출이 없어 **발화 1건당 프롬프트 1개**입니다. 배치 추적 부담이 없으므로 호스팅
  `text` 모드와 직접 비교하지 마세요. 조건은 `output_mode: local_constrained` 로 기록됩니다.

요약과 환각 벤치마크는 로컬 트랙이 지원하지 않습니다(자유 생성이 필요합니다). 해당 벤치마크를
요청하면 `SKIPPED` 로 표시하고 분류만 실행합니다.

다른 백엔드(예: GPTQ-Int8 + transformers)는 `llmbench/local/engine.py` 의 `ENGINES` 에
등록하면 됩니다. 엔진은 `classify(system, user, candidates) -> LocalResult` 하나만 구현하면
나머지(캐시, 채점, 리포트)는 그대로 동작합니다.

EN/KO 쌍은 **같은 semantic id** 를 공유합니다. 뜻이 같은 발화를 두 언어로 물어본 것이라,
두 언어 사이의 편차를 직접 볼 수 있습니다.

| 지표 | 방향 | 의미 |
|---|:--:|---|
| Cross-lingual consistency | ↑ | 두 언어에서 정오답이 일치한 비율(둘 다 맞음 + 둘 다 틀림). |
| Prediction agreement | ↑ | 정답 여부와 무관하게 **같은 인텐트** 를 예측한 비율. 정확도와 분리된 안정성 지표. |
| Both correct | ↑ | 두 언어 모두 맞힘. |
| EN only correct | ↓ | 영어만 맞힘 = 한국어가 약함. |
| KO only correct | ↓ | 한국어만 맞힘 = 영어가 약함. |
| Both wrong | ↓ | 두 언어 모두 틀림. |

Consistency 는 **둘 다 틀린 경우도 "일관됨"으로 셉니다.** 정확도가 낮은 모델이 높은
consistency 를 받을 수 있으므로 반드시 Accuracy 와 함께 보세요.

### 부가: Surface fact support (진단)

생성된 요약에서 숫자·백분율·날짜·통화·개체명을 **규칙 기반으로** 추출해 원문과 대조합니다.
LLM 을 쓰지 않는 결정적 계산입니다.

```
surface_fact_support_precision = 뒷받침되는 추출 사실 / 전체 추출 사실
```

명세의 예시대로 원문 `매출은 12% 증가` → 요약 `매출은 21% 증가` 를 잡아내고,
`number` / `percentage` / `date` / `currency` / `unsupported entity` 로 나눠 집계합니다.

**이것은 진단 지표이지 환각 지표가 아닙니다.** 의미 수준의 왜곡("부인했다" vs "확인했다")과
누락은 보지 못하고, 개체명 후보는 정밀도 위주로 적게 뽑습니다. 점수가 깨끗하다고 충실한
요약이라는 뜻이 아닙니다.

### 공통 규칙

- **파싱 불가 출력은 오답입니다.** 분모에서 빼지 않고 `invalid_output_rate` 로 따로 보고합니다.
  형식을 못 지키는 것도 모델의 능력입니다.
- **EN 과 KO 는 항상 따로 보고합니다.** 평균 내지 않습니다.
- **성격이 다른 지표를 하나의 종합 점수로 합치지 않습니다.** 요약 ROUGE 와 분류 정확도는
  같은 축이 아닙니다.
- 위 숫자 중 **어느 것도 LLM-as-a-Judge 를 쓰지 않습니다.** Judge 는 제공되지만 기본 비활성이고
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
uv sync --extra local       # 로컬 분류 실행 (llama.cpp)
uv sync --extra dev         # 테스트 의존성
```

`local` extra 의 CPU 휠은 별도 인덱스가 필요합니다.

```bash
uv sync --extra local --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
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

`structured_outputs` 는 `response_format` 과 **별개 능력**입니다. endpoint 가
`{"type": "json_object"}` 는 받으면서 strict schema 는 거부할 수 있어서, 둘을 같은 것으로
취급하면 structured output 을 지원한다고 표시된 모델이 모든 요청에 404 를 돌려줍니다.

일부 제약은 메타데이터에 드러나지 않습니다. 예를 들어 thinking 을 끌 수 없는 모델은
`supported_parameters` 만으로는 구분되지 않고, 실행해 봐야 400 이 돌아옵니다. 이런 런타임
제약은 사전 검사가 아니라 circuit breaker 가 잡습니다.
[문제 해결](#reasoning-is-mandatory-for-this-endpoint-and-cannot-be-disabled) 참고.

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
| `--classification-mode` | `text`(기본) 또는 `json_schema` |
| `--all-local` | `config/local_models.yaml` 의 로컬 모델 포함 |
| `--skip-preflight` | 모델이 거부하는 파라미터가 있어도 강행 |
| `-v` / `-vv` | 콘솔 로그를 INFO / DEBUG 로 |
| `--log-file PATH` | 로그 파일 위치 변경 |
| `--no-report` | 끝난 뒤 RESULT.md / 스프레드시트를 갱신하지 않음 |

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
RESULT.md                                   사람이 읽는 결과 문서 (자동 생성)
results/
  benchmark_results.xlsx                    같은 데이터를 담은 스프레드시트
  leaderboard.csv
  leaderboard.json
  leaderboard.md
  report.html
  models/{safe_model_slug}/
    summary.json                            기본 조건
    summary.classification-json_schema.json 구조화 출력 조건 (있을 때만)
```

`benchmark run` 도 끝날 때 같은 산출물을 자동으로 갱신합니다(`--no-report` 로 끌 수 있음).
모든 산출물의 **입력은 `results/models/*/summary.json` 하나뿐** 입니다. 모델을 하나 돌리면
그 파일이 갱신되고, 나머지는 전부 거기서 다시 만들어집니다.

raw OpenRouter 응답은 DB에만 저장하고 리포트에는 싣지 않습니다.

### RESULT.md

자동 생성 문서입니다. **직접 수정하지 마세요.** 다음 절로 구성됩니다.

리더보드 · 요약(XL-Sum) · 환각 탐지 · 분류(MASSIVE) · EN/KO 교차 일관성 ·
Surface fact support(진단) · 비용 · 토큰 사용량/캐시 · LLM-as-a-Judge(사용한 경우만) ·
재현성 · 비고(중단·건너뛴 항목).

맨 앞에 "숫자를 읽기 전에" 주의사항이 붙습니다. 이 문서는 PR이나 메신저로 그대로
복사되기 때문에, 주의사항이 숫자와 함께 따라다녀야 합니다.

### 스프레드시트

`results/benchmark_results.xlsx` 는 RESULT.md 와 **완전히 같은 데이터**를 시트로 나눈 것입니다
(두 산출물 모두 같은 테이블 빌더를 씁니다).

| 시트 | 내용 |
|---|---|
| `About` | 생성 시각, 버전, 주의사항 |
| `Leaderboard` | 리더보드 전체 컬럼 |
| `Summarization` / `Hallucination` / `Classification` | 모델 × 언어 |
| `CrossLingual` | EN/KO 교차 일관성 분해 |
| `SurfaceFacts` | 진단 지표 오류 유형별 집계 |
| `Cost` / `Usage` | 비용과 토큰, 가격 스냅샷 |
| `Judge` | judge 사용 시 |
| `Runs` | 재현성 메타데이터(git commit, manifest 해시) |

헤더 고정·필터·숫자 서식이 적용돼 있습니다.

### 자동 갱신 (GitHub Actions)

`.github/workflows/results.yml` 이 `results/models/**/summary.json` 변경을 감지해
RESULT.md 와 스프레드시트를 다시 만들고 커밋·푸시합니다.

- **입력만 트리거입니다.** 생성된 파일(`RESULT.md`, `results/leaderboard.*`,
  `results/benchmark_results.xlsx`)은 트리거 경로에 없으므로 워크플로가 자기 자신을
  다시 부르지 않습니다. `GITHUB_TOKEN` 으로 한 푸시는 워크플로를 트리거하지 않는다는
  GitHub 동작이 이중 안전장치입니다.
- 커밋 전에 `tests/test_reporting.py` 를 돌립니다. 생성기가 깨진 채로 잘못된 RESULT.md 가
  커밋되는 것을 막기 위해서입니다.
- 네트워크도 OpenRouter 키도 쓰지 않습니다. `summary.json` 만 읽습니다. 비용 0.
- 산출물은 워크플로 artifact 로도 90일간 올라갑니다.
- `workflow_dispatch` 로 수동 실행할 수 있고, 이때 `commit: false` 를 주면 커밋 없이
  artifact 만 만듭니다.

로컬에서 결과를 만들었다면 `results/models/<slug>/summary.json` 을 커밋해서 푸시하면 됩니다.
나머지는 워크플로가 처리합니다.

리더보드 컬럼은 모델, input/output 가격, 요약 EN/KO, 환각 F1 EN/KO, 분류 EN/KO,
EN-KO consistency, 토큰 합계, OpenRouter 비용, judge 비용입니다. 모델 상세에는 ROUGE-Lsum,
BERTScore, chrF++, surface fact support, compression ratio, 환각 accuracy/macro-F1/recall,
분류 accuracy/macro-F1, cross-lingual 분해가 추가됩니다.

### 숫자를 인용하기 전에 알아둘 것

각 지표가 무엇이고 어느 방향이 좋은지는 [무엇을 측정하는가](#무엇을-측정하는가)에 정리돼
있습니다. `RESULT.md` 의 각 표 아래 `지표 설명` 과 스프레드시트의 `Legend` 시트에도 같은 내용이
들어갑니다. 여기서는 **결과 파일을 읽을 때만 걸리는 함정**을 다룹니다.

**한국어 ROUGE는 문자 단위 토큰화를 씁니다.** `rouge-score` 의 기본 토크나이저는 `[a-z0-9]`
이외 문자를 모두 제거해서 한국어 텍스트를 통째로 날려버리고 모든 모델에 0.0을 줍니다. 그래서
다국어 토크나이저를 넣었습니다. 라틴 문자는 단어 단위, 한글/CJK는 문자 단위입니다. 따라서
한국어 ROUGE 값은 *이 벤치마크 안의 모델 간 비교* 에는 유효하지만, 단어 단위인 영어 ROUGE와
직접 비교할 수는 없습니다. 한국어 값이 영어보다 높게 나오는 것도 이 때문입니다.

**빈 칸은 0이 아니라 "계산되지 않음"입니다.** 리포트의 `—` 와 스프레드시트의 빈 셀은 값이
없다는 뜻입니다. 가장 흔한 경우는 BERTScore(선택 extra 미설치)와 `SKIPPED` 상태의 한국어
환각 벤치마크입니다. 절대 `0.0` 으로 채우지 않습니다.

**thinking 을 켠 모델은 조건이 다릅니다.** thinking 을 끌 수 없어 켜 둔 모델은 reasoning token 이
추가 과금되고 출력 길이 상한도 다릅니다. `Reasoning` 토큰 열이 0이 아닌 모델이 여기 해당합니다.
[문제 해결](#reasoning-is-mandatory-for-this-endpoint-and-cannot-be-disabled) 참고.

**비용은 마지막 실행이 아니라 누적입니다.** `Total` 은 그 모델의 캐시를 만드는 데 지금까지
지불한 총액이고, 이번 실행에서 새로 나간 금액은 `Fresh this run` 입니다. 캐시가 다 맞으면
후자는 $0 입니다.

**주의사항 세 가지는 어디서 보든 따라옵니다.** 종합 점수를 만들지 않는다는 것,
Surface fact support 가 진단 지표라는 것, EN/KO 환각 점수가 직접 비교 대상이 아니라는 것은
RESULT.md 맨 앞과 스프레드시트 `About` 시트에도 들어갑니다. 표만 떼어가도 맥락이 남도록
한 것입니다.

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

### `Reasoning is mandatory for this endpoint and cannot be disabled`

일부 모델은 thinking 을 끌 수 없습니다. 기본값이 `reasoning.enabled: false` 이므로 이런 모델은
모든 요청이 400 으로 실패하고, circuit breaker 가 10회 만에 중단시킵니다.

해당 모델만 thinking 을 켜되 가장 낮은 강도로 두고, **출력 여유분을 주세요.**

```yaml
- model_id: openai/gpt-5-nano
  max_output_tokens: 4096
  reasoning:
    enabled: true
    effort: low                    # 명세의 "disabled 또는 최소화" 중 최소화 쪽
    output_token_headroom: 2048    # reasoning token 은 max_tokens 에 함께 계산됨
    estimated_output_tokens: 256   # 비용 추정 전용 (headroom 은 상한일 뿐)
```

`output_token_headroom` 이 핵심입니다. 환각 벤치마크의 답변 예산은 **8 토큰** 이라,
여유분이 없으면 모델이 생각할 공간 자체가 없어 빈 응답이 나옵니다. 실측에서
`openai/gpt-5-nano` 는 환각 케이스 하나에 reasoning 만 384 토큰을 썼습니다.

`openai/gpt-5-nano` 와 `z-ai/glm-5.3-flash` 는 이미 이렇게 설정돼 있습니다.

**알아둘 점 두 가지입니다.**

- **비교 조건이 달라집니다.** thinking 을 켠 모델은 reasoning token 이 추가 과금되고
  출력 특성도 달라집니다. `summary.json` 의 `reproducibility.reasoning_config` 와
  리포트의 `Reasoning Tokens` 열에 기록됩니다.
- **출력 길이 상한도 달라집니다.** 비추론 모델은 요약에서 200 토큰에서 잘리지만, 여유분을
  받은 모델은 상한이 훨씬 높습니다. 프롬프트가 1~3문장을 지시하므로 실제 길이는 비슷하지만,
  길이 상한이 동일하지 않다는 점은 감안하세요.

### `The specified schema produces a constraint that has too many states for serving`

Gemini 계열이 structured output 요청에 400 으로 응답할 때 나옵니다. 제공자가 스키마를
제약 디코딩 상태 기계로 컴파일하는데, **enum 값이 많은 배열에 길이 제약을 함께 걸면** 한계를
넘습니다(60개 인텐트 × `minItems/maxItems: 20`).

그래서 분류 스키마는 배열 길이를 제약하지 않습니다. 길이는 프롬프트가 지시하고
파서가 검사하며, 개수가 맞지 않으면 그 배치 전체를 invalid 로 처리합니다. 정렬 보장은
디코딩 시점이 아니라 채점 시점에 이뤄지고, 길이 실패는 `invalid_output_rate` 에 정직하게
드러납니다.

### structured output 인데 404 가 난다

모델 단위 `supported_parameters` 는 endpoint 들의 **합집합**이라 낙관적입니다. 실제로 그
스키마를 처리할 endpoint 가 없으면 `require_parameters: true` 에서 404 가 납니다.
`https://openrouter.ai/api/v1/models/<slug>/endpoints` 로 endpoint별 지원을 확인하고,
해당 모델에 `parameters.structured_outputs: false` 를 설정하세요. 그러면 사전 검사가
`json_schema` 모드 실행을 **요청 전에** 막고, `text` 모드는 정상 동작합니다.

### 로컬 실행이 느리다 / `rejects prefix reuse` 로그가 보인다

PCD 는 발화마다 같은 시스템 메시지(인텐트 60개)를 다시 prefill 하지 않으려고, 공유 접두부를
KV 캐시에 두고 달라지는 뒷부분만 되감아 재평가합니다.

멀티모달 RoPE 모델(Qwen3.5 계열)은 이를 거부합니다. 시퀀스의 위치가 **엄격히 증가**해야 해서,
되감아 비운 위치에서 다시 시작하는 배치를 llama.cpp 가 받지 않습니다. 이 경우 엔진은 재사용을
포기하고 매번 전체 prefill 합니다. 로그에 한 번 기록되고, 이후에는 조용히 느려집니다.

접두부를 **새 시퀀스로 복사**하면 위치 검사는 통과하지만 **답이 틀려집니다.** M-RoPE 위치는
복사로 복원되는 스칼라가 아니라서, 오류 없이 정확도만 조용히 떨어집니다(실측: Qwen3.5-4B 가
같은 발화 80건에서 0.838 → 0.750). 그래서 그 경로는 쓰지 않습니다. 느리고 맞는 쪽이
빠르고 조용히 틀린 쪽보다 낫습니다.

### `ran out of KV sequences` / `branches need evaluating`

Parallel Constrained Decoding 이 한 깊이에서 동시에 평가해야 할 분기가
`runtime.n_seq_max` 를 넘었습니다. 후보 집합이 크거나 토큰 접두부가 많이 겹칠 때 생깁니다.
값을 올리세요(라이브러리 상한 256). 조용히 잘라내면 argmax 가 달라질 수 있어 일부러
오류로 처리합니다.

### 로컬 실행이 출력 없이 종료된다 (exit 132 / SIGILL)

`llama-cpp-python` 사전 빌드 휠 중 일부는 **AVX-512** 로 빌드돼 있어, 이를 지원하지 않는 CPU
(예: Coffee Lake i9-9900K)에서는 import 시점에 SIGILL 로 죽습니다. 파이썬 예외가 아니라
프로세스 종료라 아무 메시지도 남지 않습니다.

```bash
grep -o -E "avx512[a-z0-9_]*" /proc/cpuinfo | sort -u   # 비어 있으면 AVX-512 없음
```

다른 `cuXXX` 태그를 시도하거나(이 저장소는 `cu122` 로 검증했습니다) 소스에서 빌드하세요.

### 로컬 실행이 모델 로딩 중 종료된다

VRAM 부족입니다. `nvidia-smi` 로 여유 메모리를 확인하고 `n_gpu_layers` 를 낮추세요.
데스크톱 환경이 이미 2GB 이상 쓰고 있는 경우가 흔합니다.

### 실행이 `ABORTED` 로 끝났다
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
  local/         로컬 실행 백엔드 (분류 전용, OpenRouter 아님)
                 pcd_engine.py = Parallel Constrained Decoding
                 llama_cpp_engine.py = 문법 유도 생성 (baseline)
  reporting/     모델별 summary, 리더보드, HTML, RESULT.md, Excel
                 (tables.py 가 두 산출물의 공통 테이블 정의)
  runner.py      plan -> run -> score -> persist
  budget.py      추정과 지출 가드
  failures.py    실패 중단 (circuit breaker)
  logging_setup.py  콘솔/파일 로깅
  pricing.py     들어올 땐 추정치, 나갈 땐 usage.cost
  cli.py         `benchmark` 커맨드
tests/           단위 테스트 + mock transport 기반 end-to-end 벤치마크
scripts/         전체 규모 mock end-to-end 데모
.github/workflows/results.yml   결과 문서 자동 갱신
```

**API provider adapter 는 정확히 하나입니다.** `OpenAIProvider`, `GoogleProvider`,
`MistralProvider`, `AnthropicProvider` 같은 클래스는 없으며, 벤치마크 코어가 vendor를 보고
endpoint를 분기하는 일도 없습니다.

`local/` 은 두 번째 vendor 연동이 아니라 **다른 실행 백엔드**입니다. 모델 API 와 통신하는 것은
여전히 `OpenRouterClient` 하나뿐이고, 로컬 엔진은 아무 API 도 호출하지 않습니다. 러너가
그대로 재사용되도록 같은 client 인터페이스를 구현할 뿐입니다.

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
