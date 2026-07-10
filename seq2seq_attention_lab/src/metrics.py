"""Metricas de avaliacao para sequencias."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence


@dataclass
class SequenceMetrics:
    expected: List[int]
    predicted: List[int]
    correct_tokens: int
    total_expected_tokens: int
    token_accuracy: float
    exact_match: bool
    extra_tokens: int
    missing_tokens: int


def calculate_sequence_metrics(expected: Sequence[int], predicted: Sequence[int]) -> SequenceMetrics:
    """
    Calcula metricas para uma sequencia.

    token_accuracy segue a formula:
    tokens_corretos / total_de_tokens_esperados
    """
    expected_list = list(expected)
    predicted_list = list(predicted)

    total_expected = len(expected_list)
    correct = 0

    for index in range(total_expected):
        if index < len(predicted_list) and predicted_list[index] == expected_list[index]:
            correct += 1

    accuracy = correct / total_expected if total_expected else 0.0

    extra = max(0, len(predicted_list) - len(expected_list))
    missing = max(0, len(expected_list) - len(predicted_list))

    return SequenceMetrics(
        expected=expected_list,
        predicted=predicted_list,
        correct_tokens=correct,
        total_expected_tokens=total_expected,
        token_accuracy=accuracy,
        exact_match=expected_list == predicted_list,
        extra_tokens=extra,
        missing_tokens=missing,
    )


def summarize_metrics(metrics: Sequence[SequenceMetrics]) -> dict:
    """Resume uma lista de metricas de sequencia."""
    if not metrics:
        return {
            "avg_token_accuracy": 0.0,
            "exact_match_rate": 0.0,
            "avg_extra_tokens": 0.0,
            "avg_missing_tokens": 0.0,
        }

    n = len(metrics)
    return {
        "avg_token_accuracy": sum(item.token_accuracy for item in metrics) / n,
        "exact_match_rate": sum(1 for item in metrics if item.exact_match) / n,
        "avg_extra_tokens": sum(item.extra_tokens for item in metrics) / n,
        "avg_missing_tokens": sum(item.missing_tokens for item in metrics) / n,
    }
