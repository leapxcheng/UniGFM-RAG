"""QA evaluation metrics for measuring answer quality.

This module provides standard QA evaluation metrics including:
- Exact Match (EM): Normalized string equality
- Fuzzy Match (FM): Containment-based matching
- Token F1: Token-level precision/recall/F1

These metrics can be used standalone or registered for use in trainers.
"""

import re
import string
from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Callable

__all__ = [
    "normalize_answer",
    "compute_exact_match",
    "compute_fuzzy_match",
    "compute_f1",
    "extract_answer_from_response",
    "QAMetric",
    "ExactMatchMetric",
    "FuzzyMatchMetric",
    "TokenF1Metric",
    "QA_METRICS_REGISTRY",
    "get_qa_metric",
]


# =====================================================
# Answer Normalization
# =====================================================


def normalize_answer(s: str) -> str:
    """Normalize answer text for evaluation.

    Performs lowercase conversion, punctuation removal,
    article removal, and whitespace normalization.

    Args:
        s: Input string to normalize

    Returns:
        Normalized string
    """

    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


# =====================================================
# Metric Functions
# =====================================================


def compute_exact_match(prediction: str, ground_truth: str) -> float:
    """Compute exact match (EM) score.

    Returns 1.0 if normalized prediction equals normalized ground truth,
    0.0 otherwise.

    Args:
        prediction: Predicted answer string
        ground_truth: Ground truth answer string

    Returns:
        1.0 if exact match, 0.0 otherwise
    """
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def compute_f1(prediction: str, ground_truth: str) -> float:
    """Compute token-level F1 score.

    Computes F1 based on token overlap between normalized prediction
    and normalized ground truth.

    Args:
        prediction: Predicted answer string
        ground_truth: Ground truth answer string

    Returns:
        Token-level F1 score between 0.0 and 1.0
    """
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    # Handle special cases for yes/no/noanswer
    if (
        normalized_prediction in ["yes", "no", "noanswer"]
        and normalized_prediction != normalized_ground_truth
    ):
        return 0.0
    if (
        normalized_ground_truth in ["yes", "no", "noanswer"]
        and normalized_prediction != normalized_ground_truth
    ):
        return 0.0

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()

    if len(prediction_tokens) == 0 or len(ground_truth_tokens) == 0:
        return float(prediction_tokens == ground_truth_tokens)

    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = num_same / len(prediction_tokens)
    recall = num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)

    return f1


def compute_fuzzy_match(prediction: str, ground_truth: str) -> float:
    """Compute fuzzy match (FM) score.

    Returns 1.0 if normalized ground truth is contained in normalized prediction
    or vice versa, 0.0 otherwise.

    Args:
        prediction: Predicted answer string
        ground_truth: Ground truth answer string

    Returns:
        1.0 if fuzzy match, 0.0 otherwise
    """
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    # Check if either is contained in the other
    if normalized_ground_truth in normalized_prediction:
        return 1.0
    if normalized_prediction in normalized_ground_truth:
        return 1.0

    return 0.0


def extract_answer_from_response(response: str) -> str:
    """Extract answer from model response.

    Handles common patterns like "Answer: <answer>" or just returns
    the full response if no pattern is found.

    Args:
        response: Full model response string

    Returns:
        Extracted answer string
    """
    # Try to extract answer after "Answer:" marker
    if "Answer:" in response:
        return response.split("Answer:")[-1].strip()
    if "answer:" in response.lower():
        # Case-insensitive search
        parts = re.split(r"answer:", response, flags=re.IGNORECASE)
        if len(parts) > 1:
            return parts[-1].strip()

    # Return full response if no pattern found
    return response.strip()


# =====================================================
# Metric Classes (for registry-based usage)
# =====================================================


class QAMetric(ABC):
    """Base class for QA evaluation metrics."""

    name: str = "base"

    @abstractmethod
    def __call__(self, prediction: str, ground_truth: str) -> float:
        """Compute the metric score.

        Args:
            prediction: Predicted answer string
            ground_truth: Ground truth answer string

        Returns:
            Metric score (typically between 0.0 and 1.0)
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name})"


class ExactMatchMetric(QAMetric):
    """Exact Match (EM) metric."""

    name: str = "em"

    def __call__(self, prediction: str, ground_truth: str) -> float:
        return compute_exact_match(prediction, ground_truth)


class FuzzyMatchMetric(QAMetric):
    """Fuzzy Match (FM) metric - containment-based matching."""

    name: str = "fm"

    def __call__(self, prediction: str, ground_truth: str) -> float:
        return compute_fuzzy_match(prediction, ground_truth)


class TokenF1Metric(QAMetric):
    """Token-level F1 metric."""

    name: str = "f1"

    def __call__(self, prediction: str, ground_truth: str) -> float:
        return compute_f1(prediction, ground_truth)


# =====================================================
# Metric Registry
# =====================================================


# Registry mapping metric names to their classes/functions
QA_METRICS_REGISTRY: dict[str, type[QAMetric] | Callable[[str, str], float]] = {
    "em": ExactMatchMetric,
    "exact_match": ExactMatchMetric,
    "fm": FuzzyMatchMetric,
    "fuzzy_match": FuzzyMatchMetric,
    "f1": TokenF1Metric,
    "token_f1": TokenF1Metric,
}


def get_qa_metric(name: str) -> QAMetric:
    """Get a QA metric by name from the registry.

    Args:
        name: Metric name (e.g., "em", "fm", "f1")

    Returns:
        Instantiated QAMetric object

    Raises:
        ValueError: If metric name is not found in registry
    """
    name_lower = name.lower()
    if name_lower not in QA_METRICS_REGISTRY:
        available = list(QA_METRICS_REGISTRY.keys())
        raise ValueError(f"Unknown QA metric: {name}. Available metrics: {available}")

    metric_cls = QA_METRICS_REGISTRY[name_lower]
    if isinstance(metric_cls, type):
        return metric_cls()
    else:
        # For backward compatibility with function-based metrics
        class FunctionMetric(QAMetric):
            name = name_lower

            def __call__(self, prediction: str, ground_truth: str) -> float:
                return metric_cls(prediction, ground_truth)

        return FunctionMetric()


def register_qa_metric(
    name: str, metric: type[QAMetric] | Callable[[str, str], float]
) -> None:
    """Register a new QA metric.

    Args:
        name: Name to register the metric under
        metric: Metric class or function
    """
    QA_METRICS_REGISTRY[name.lower()] = metric
