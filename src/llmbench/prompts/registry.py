"""Versioned prompt templates.

Every template carries a version string that participates in the inference cache
key, so editing a prompt correctly invalidates exactly the affected cases and
nothing else.

The EN and KO variants are intentionally parallel: the task, the constraints and
the output contract are the same, so the EN/KO comparison measures the model, not
a difference in instructions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

__all__ = [
    "PROMPT_VERSIONS",
    "RenderedPrompt",
    "summarization_prompt",
    "hallucination_prompt",
    "classification_prompt",
    "classification_json_schema",
    "judge_prompt",
]

PROMPT_VERSIONS = {
    "summarization": "sum-1",
    "hallucination": "halu-1",
    "classification": "cls-1",
    # Structured-output variant: the model returns a JSON array constrained to the
    # frozen label space instead of writing `<n>: <intent number>` by hand.
    "classification_json_schema": "cls-json-1",
    "judge": "judge-1",
}


@dataclass(frozen=True, slots=True)
class RenderedPrompt:
    system: str
    user: str
    template_version: str

    @property
    def messages(self) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": self.system},
            {"role": "user", "content": self.user},
        ]


# --------------------------------------------------------------------------- #
# Summarization
# --------------------------------------------------------------------------- #
_SUM_SYSTEM = {
    "en": (
        "You are a precise news summarizer. You write faithful, self-contained "
        "summaries and never add information that is not in the source."
    ),
    "ko": (
        "당신은 정확한 뉴스 요약가입니다. 원문에 없는 정보를 절대 추가하지 않고, "
        "원문에 충실한 요약문을 작성합니다."
    ),
}

_SUM_USER = {
    "en": (
        "Summarize the news article below in English.\n\n"
        "Rules:\n"
        "- Write 1 to 3 sentences.\n"
        "- Use only facts stated in the article.\n"
        "- Do not add opinions, headings, bullet points or any preamble.\n"
        "- Output the summary text only.\n\n"
        "Article:\n{source}\n\n"
        "Summary:"
    ),
    "ko": (
        "아래 뉴스 기사를 한국어로 요약하라.\n\n"
        "규칙:\n"
        "- 1~3문장으로 작성한다.\n"
        "- 기사에 명시된 사실만 사용한다.\n"
        "- 의견, 제목, 목록 기호, 머리말을 붙이지 않는다.\n"
        "- 요약문만 출력한다.\n\n"
        "기사:\n{source}\n\n"
        "요약:"
    ),
}


def summarization_prompt(language: str, source: str) -> RenderedPrompt:
    lang = language if language in _SUM_USER else "en"
    return RenderedPrompt(
        system=_SUM_SYSTEM[lang],
        user=_SUM_USER[lang].format(source=source),
        template_version=PROMPT_VERSIONS["summarization"],
    )


# --------------------------------------------------------------------------- #
# Hallucination / factual consistency (wording fixed by the spec, section 8)
# --------------------------------------------------------------------------- #
_HALU_SYSTEM = {
    "en": "You are a strict factual-consistency checker. You answer with one label and nothing else.",
    "ko": "당신은 엄격한 사실 일치 판정기다. 오직 하나의 레이블만 출력한다.",
}

_HALU_USER = {
    "en": (
        "Read the source document and candidate summary.\n\n"
        "Determine whether every factual claim in the candidate summary\n"
        "is supported by the source document.\n\n"
        "Return exactly one label:\n\n"
        "SUPPORTED\n"
        "HALLUCINATED\n\n"
        "Source document:\n{source}\n\n"
        "Candidate summary:\n{candidate}\n\n"
        "Label:"
    ),
    "ko": (
        "원문과 후보 요약문을 읽고,\n"
        "후보 요약문의 모든 사실적 주장이 원문에 의해 뒷받침되는지 판단하라.\n\n"
        "다음 중 하나만 출력하라.\n\n"
        "SUPPORTED\n"
        "HALLUCINATED\n\n"
        "원문:\n{source}\n\n"
        "후보 요약문:\n{candidate}\n\n"
        "레이블:"
    ),
}


def hallucination_prompt(language: str, source: str, candidate: str) -> RenderedPrompt:
    lang = language if language in _HALU_USER else "en"
    return RenderedPrompt(
        system=_HALU_SYSTEM[lang],
        user=_HALU_USER[lang].format(source=source, candidate=candidate),
        template_version=PROMPT_VERSIONS["hallucination"],
    )


# --------------------------------------------------------------------------- #
# Classification (batched: N utterances per OpenRouter request)
# --------------------------------------------------------------------------- #
_CLS_SYSTEM = {
    "en": (
        "You are an intent classifier for a voice assistant. You answer with one "
        "line per input, in the exact format requested, and nothing else."
    ),
    "ko": (
        "당신은 음성 비서의 인텐트 분류기다. 요청된 형식 그대로 입력 하나당 한 줄씩만 "
        "출력하고, 그 외에는 아무것도 출력하지 않는다."
    ),
}

_CLS_USER = {
    "en": (
        "Classify each user utterance into exactly one intent from the list below.\n\n"
        "Intents:\n{labels}\n\n"
        "Utterances:\n{items}\n\n"
        "Output format: one line per utterance, `<utterance number>: <intent number>`.\n"
        "Output exactly {count} lines, numbered 1 to {count}, and nothing else.\n\n"
        "Answer:"
    ),
    "ko": (
        "아래 목록에서 각 사용자 발화에 해당하는 인텐트를 정확히 하나 선택하라.\n\n"
        "인텐트 목록:\n{labels}\n\n"
        "발화:\n{items}\n\n"
        "출력 형식: 발화 하나당 한 줄로 `<발화 번호>: <인텐트 번호>`.\n"
        "정확히 {count}줄을 1부터 {count}까지 번호 순으로 출력하고, 그 외에는 아무것도 출력하지 마라.\n\n"
        "답:"
    ),
}


def format_label_space(label_space: Sequence[str]) -> str:
    """1-based numbering of the frozen label space, identical for EN and KO."""
    return "\n".join(f"{i}: {label}" for i, label in enumerate(label_space, start=1))


# --- structured-output variant ------------------------------------------- #
# With `response_format`, the answer is constrained to the label space by the
# provider, so the prompt asks for intent *names* and drops the numbering
# indirection: an off-by-one in a number list is invisible, a wrong name is not.
_CLS_JSON_SYSTEM = {
    "en": (
        "You are an intent classifier for a voice assistant. You reply with a single "
        "JSON object matching the given schema, and nothing else."
    ),
    "ko": (
        "당신은 음성 비서의 인텐트 분류기다. 주어진 스키마에 맞는 JSON 객체 하나만 "
        "출력하고, 그 외에는 아무것도 출력하지 않는다."
    ),
}

_CLS_JSON_USER = {
    "en": (
        "Classify each user utterance into exactly one intent from the list below.\n\n"
        "Intents:\n{labels}\n\n"
        "Utterances:\n{items}\n\n"
        'Reply with exactly this JSON object: {{"answers": ["<intent for utterance 1>", '
        '"<intent for utterance 2>", ...]}}\n'
        "The `answers` array must contain exactly {count} intent names, in the same order "
        "as the utterances above.\n\n"
        "Answer:"
    ),
    "ko": (
        "아래 목록에서 각 사용자 발화에 해당하는 인텐트를 정확히 하나 선택하라.\n\n"
        "인텐트 목록:\n{labels}\n\n"
        "발화:\n{items}\n\n"
        '정확히 다음 JSON 객체만 출력하라: {{"answers": ["<1번 발화의 인텐트>", '
        '"<2번 발화의 인텐트>", ...]}}\n'
        "`answers` 배열에는 위 발화와 같은 순서로 정확히 {count}개의 인텐트 이름이 들어가야 한다.\n\n"
        "답:"
    ),
}


def classification_json_schema(label_space: Sequence[str], count: int) -> dict[str, Any]:
    """OpenRouter `response_format` constraining every answer to the label space.

    `enum` is what makes an unparseable answer structurally impossible on a
    provider that honours strict structured outputs.

    The array length is deliberately NOT constrained with `minItems`/`maxItems`.
    Providers compile the schema into a constrained-decoding state machine, and
    a 60-value enum combined with a fixed array length of 20 exceeds Gemini's
    limit outright ("The specified schema produces a constraint that has too
    many states for serving"). The prompt still asks for exactly ``count``
    answers, and `parse_batch_json_answer` invalidates the whole batch on a
    length mismatch -- so alignment is still guaranteed, it is just enforced at
    scoring time instead of decoding time, and a bad length shows up honestly in
    `invalid_output_rate`.
    """
    del count  # length is enforced by the prompt and the parser, not the schema
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "intent_classification",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "answers": {
                        "type": "array",
                        "items": {"type": "string", "enum": list(label_space)},
                    }
                },
                "required": ["answers"],
                "additionalProperties": False,
            },
        },
    }


def classification_prompt(
    language: str,
    items: Sequence[str],
    label_space: Sequence[str],
    *,
    structured: bool = False,
) -> RenderedPrompt:
    lang = language if language in _CLS_USER else "en"
    numbered = "\n".join(f"{i}: {text}" for i, text in enumerate(items, start=1))
    system = (_CLS_JSON_SYSTEM if structured else _CLS_SYSTEM)[lang]
    template = (_CLS_JSON_USER if structured else _CLS_USER)[lang]
    labels = (
        "\n".join(f"- {label}" for label in label_space)
        if structured
        else format_label_space(label_space)
    )
    return RenderedPrompt(
        system=system,
        user=template.format(labels=labels, items=numbered, count=len(items)),
        template_version=PROMPT_VERSIONS[
            "classification_json_schema" if structured else "classification"
        ],
    )


# --------------------------------------------------------------------------- #
# Optional judge
# --------------------------------------------------------------------------- #
_JUDGE_RUBRIC = {
    "en": (
        "Score each dimension from 1 to 5 (integers only):\n"
        "- factual_consistency: 5 = every claim is supported by the source; "
        "1 = the summary contradicts or invents core facts.\n"
        "- key_information_coverage: 5 = captures all essential points of the "
        "reference summary; 1 = misses the main point.\n"
        "- conciseness: 5 = no redundancy or filler; 1 = padded or repetitive.\n"
        "- overall_quality: 5 = publishable as-is; 1 = unusable."
    ),
    "ko": (
        "각 항목을 1~5 정수로 채점하라.\n"
        "- factual_consistency: 5 = 모든 주장이 원문에 의해 뒷받침됨, "
        "1 = 핵심 사실을 왜곡하거나 지어냄.\n"
        "- key_information_coverage: 5 = 참조 요약의 핵심을 모두 담음, 1 = 핵심을 놓침.\n"
        "- conciseness: 5 = 군더더기 없음, 1 = 중복·장황함.\n"
        "- overall_quality: 5 = 그대로 사용 가능, 1 = 사용 불가."
    ),
}

_JUDGE_SYSTEM = {
    "en": "You are a strict summary evaluator. You reply with a single JSON object and nothing else.",
    "ko": "당신은 엄격한 요약 평가자다. 오직 하나의 JSON 객체만 출력한다.",
}

_JUDGE_USER = {
    "en": (
        "Evaluate the generated summary against the source document and the "
        "reference summary.\n\n"
        "{rubric}\n\n"
        "Source document:\n{source}\n\n"
        "Reference summary:\n{reference}\n\n"
        "Generated summary:\n{generated}\n\n"
        'Reply with exactly this JSON object: {{"factual_consistency": <1-5>, '
        '"key_information_coverage": <1-5>, "conciseness": <1-5>, '
        '"overall_quality": <1-5>, "rationale": "<one short sentence>"}}'
    ),
    "ko": (
        "원문과 참조 요약을 기준으로 생성된 요약을 평가하라.\n\n"
        "{rubric}\n\n"
        "원문:\n{source}\n\n"
        "참조 요약:\n{reference}\n\n"
        "생성된 요약:\n{generated}\n\n"
        '정확히 다음 JSON 객체만 출력하라: {{"factual_consistency": <1-5>, '
        '"key_information_coverage": <1-5>, "conciseness": <1-5>, '
        '"overall_quality": <1-5>, "rationale": "<한 문장>"}}'
    ),
}

JUDGE_JSON_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "summary_judgement",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "factual_consistency": {"type": "integer", "minimum": 1, "maximum": 5},
                "key_information_coverage": {"type": "integer", "minimum": 1, "maximum": 5},
                "conciseness": {"type": "integer", "minimum": 1, "maximum": 5},
                "overall_quality": {"type": "integer", "minimum": 1, "maximum": 5},
                "rationale": {"type": "string"},
            },
            "required": [
                "factual_consistency",
                "key_information_coverage",
                "conciseness",
                "overall_quality",
                "rationale",
            ],
            "additionalProperties": False,
        },
    },
}


def judge_prompt(language: str, source: str, reference: str, generated: str) -> RenderedPrompt:
    lang = language if language in _JUDGE_USER else "en"
    return RenderedPrompt(
        system=_JUDGE_SYSTEM[lang],
        user=_JUDGE_USER[lang].format(
            rubric=_JUDGE_RUBRIC[lang],
            source=source,
            reference=reference,
            generated=generated,
        ),
        template_version=PROMPT_VERSIONS["judge"],
    )
