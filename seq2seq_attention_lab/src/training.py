"""Rotinas de treino, avaliacao e benchmark."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence

import torch
import torch.nn as nn
import torch.optim as optim

from src.data import EOS, PAD, VOCAB_SIZE, generate_gru_batch, generate_transformer_batch
from src.metrics import SequenceMetrics, calculate_sequence_metrics, summarize_metrics
from src.models.gru_seq2seq import GRUSeq2Seq, GRUSeq2SeqConfig, predict_gru
from src.models.transformer_seq2seq import TransformerSeq2Seq, TransformerSeq2SeqConfig, predict_transformer
from src.utils import count_trainable_parameters, get_cuda_peak_memory_mb, reset_cuda_peak_memory, set_seed


@dataclass
class CommonTrainConfig:
    min_len: int = 3
    max_len: int = 7
    batch_size: int = 128
    epochs: int = 10
    batches_per_epoch: int = 100
    learning_rate: float = 0.001
    eos_weight: float = 3.0
    seed: int = 42


@dataclass
class GRUTrainConfig(CommonTrainConfig):
    embed_size: int = 32
    hidden_size: int = 64
    teacher_forcing_ratio: float = 0.1


@dataclass
class TransformerTrainConfig(CommonTrainConfig):
    d_model: int = 16
    nhead: int = 2
    dim_feedforward: int = 64
    num_encoder_layers: int = 2
    num_decoder_layers: int = 2
    dropout: float = 0.1
    max_seq_len: int = 64


def build_criterion(eos_weight: float, device: torch.device) -> nn.CrossEntropyLoss:
    """Cria CrossEntropyLoss ignorando PAD e ponderando EOS."""
    weights = torch.ones(VOCAB_SIZE, device=device)
    weights[EOS] = float(eos_weight)
    return nn.CrossEntropyLoss(weight=weights, ignore_index=PAD)


def train_gru(
    config: GRUTrainConfig,
    device: torch.device,
    progress_callback: Callable[[int, int, float], None] | None = None,
) -> Dict:
    """Treina o modelo Seq2Seq com GRU."""
    set_seed(config.seed)
    reset_cuda_peak_memory(device)

    model = GRUSeq2Seq(
        GRUSeq2SeqConfig(
            embed_size=config.embed_size,
            hidden_size=config.hidden_size,
        )
    ).to(device)

    criterion = build_criterion(config.eos_weight, device)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    loss_history: List[float] = []
    start = time.perf_counter()

    for epoch in range(1, config.epochs + 1):
        model.train()
        total_loss = 0.0

        for _ in range(config.batches_per_epoch):
            src, tgt = generate_gru_batch(
                batch_size=config.batch_size,
                min_len=config.min_len,
                max_len=config.max_len,
                device=device,
            )

            optimizer.zero_grad()
            outputs = model(src, tgt, config.teacher_forcing_ratio)
            loss = criterion(outputs.reshape(-1, VOCAB_SIZE), tgt.reshape(-1))
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())

        avg_loss = total_loss / config.batches_per_epoch
        loss_history.append(avg_loss)

        if progress_callback is not None:
            progress_callback(epoch, config.epochs, avg_loss)

    train_seconds = time.perf_counter() - start

    return {
        "name": "Seq2Seq + GRU",
        "model": model,
        "loss_history": loss_history,
        "final_loss": loss_history[-1] if loss_history else None,
        "train_seconds": train_seconds,
        "parameters": count_trainable_parameters(model),
        "peak_memory_mb": get_cuda_peak_memory_mb(device),
        "config": config,
    }


def train_transformer(
    config: TransformerTrainConfig,
    device: torch.device,
    progress_callback: Callable[[int, int, float], None] | None = None,
) -> Dict:
    """Treina o Transformer encoder-decoder."""
    set_seed(config.seed)
    reset_cuda_peak_memory(device)

    model = TransformerSeq2Seq(
        TransformerSeq2SeqConfig(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            num_encoder_layers=config.num_encoder_layers,
            num_decoder_layers=config.num_decoder_layers,
            dropout=config.dropout,
            max_seq_len=config.max_seq_len,
        )
    ).to(device)

    criterion = build_criterion(config.eos_weight, device)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    loss_history: List[float] = []
    start = time.perf_counter()

    for epoch in range(1, config.epochs + 1):
        model.train()
        total_loss = 0.0

        for _ in range(config.batches_per_epoch):
            src, decoder_input, decoder_target = generate_transformer_batch(
                batch_size=config.batch_size,
                min_len=config.min_len,
                max_len=config.max_len,
                device=device,
            )

            optimizer.zero_grad()
            outputs = model(src, decoder_input)
            loss = criterion(outputs.reshape(-1, VOCAB_SIZE), decoder_target.reshape(-1))
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())

        avg_loss = total_loss / config.batches_per_epoch
        loss_history.append(avg_loss)

        if progress_callback is not None:
            progress_callback(epoch, config.epochs, avg_loss)

    train_seconds = time.perf_counter() - start

    return {
        "name": "Transformer encoder-decoder",
        "model": model,
        "loss_history": loss_history,
        "final_loss": loss_history[-1] if loss_history else None,
        "train_seconds": train_seconds,
        "parameters": count_trainable_parameters(model),
        "peak_memory_mb": get_cuda_peak_memory_mb(device),
        "config": config,
    }


def evaluate_gru_model(
    model: GRUSeq2Seq,
    tests: Sequence[Sequence[int]],
    device: torch.device,
) -> Dict:
    return evaluate_predictions(
        model=model,
        tests=tests,
        device=device,
        predictor=lambda m, digits, d: predict_gru(m, list(digits), d),
    )


def evaluate_transformer_model(
    model: TransformerSeq2Seq,
    tests: Sequence[Sequence[int]],
    device: torch.device,
) -> Dict:
    return evaluate_predictions(
        model=model,
        tests=tests,
        device=device,
        predictor=lambda m, digits, d: predict_transformer(m, list(digits), d),
    )


def evaluate_predictions(
    model: torch.nn.Module,
    tests: Sequence[Sequence[int]],
    device: torch.device,
    predictor: Callable[[torch.nn.Module, Sequence[int], torch.device], List[int]],
) -> Dict:
    """Avalia modelo nos testes e mede tempo medio de inferencia."""
    model.eval()

    results: List[SequenceMetrics] = []
    inference_times: List[float] = []

    for test in tests:
        expected = list(reversed(list(test)))

        start = time.perf_counter()
        prediction = predictor(model, test, device)
        inference_times.append(time.perf_counter() - start)

        results.append(calculate_sequence_metrics(expected, prediction))

    summary = summarize_metrics(results)
    summary["avg_inference_ms"] = (sum(inference_times) / len(inference_times) * 1000) if inference_times else 0.0
    summary["total_inference_ms"] = sum(inference_times) * 1000

    return {
        "sequence_metrics": results,
        "summary": summary,
    }
