from llmbench.benchmarks.base import Task, TaskGroup
from llmbench.benchmarks.classification import build_classification_tasks, score_classification
from llmbench.benchmarks.hallucination import build_hallucination_tasks, score_hallucination
from llmbench.benchmarks.summarization import build_summarization_tasks, score_summarization

__all__ = [
    "Task",
    "TaskGroup",
    "build_summarization_tasks",
    "score_summarization",
    "build_hallucination_tasks",
    "score_hallucination",
    "build_classification_tasks",
    "score_classification",
]

BENCHMARK_NAMES = ("summarization", "hallucination", "classification")
