"""Modelo Sequence-to-Sequence classico com GRU."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn

from src.data import DIGIT_OFFSET, EOS, PAD, SOS, VOCAB_SIZE, decode_tokens_int


@dataclass
class GRUSeq2SeqConfig:
    embed_size: int = 32
    hidden_size: int = 64


class EncoderGRU(nn.Module):
    """Encoder recorrente que comprime a entrada em um hidden state final."""

    def __init__(self, vocab_size: int, embed_size: int, hidden_size: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=PAD)
        self.gru = nn.GRU(embed_size, hidden_size, batch_first=True)

    def forward(self, src: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedded = self.embedding(src)
        outputs, hidden = self.gru(embedded)
        return outputs, hidden


class DecoderGRU(nn.Module):
    """Decoder recorrente que gera a saida token por token."""

    def __init__(self, vocab_size: int, embed_size: int, hidden_size: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=PAD)
        self.gru = nn.GRU(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_token: torch.Tensor, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        input_token = input_token.unsqueeze(1)
        embedded = self.embedding(input_token)
        output, hidden = self.gru(embedded, hidden)
        logits = self.fc(output.squeeze(1))
        return logits, hidden


class GRUSeq2Seq(nn.Module):
    """
    Modelo Seq2Seq com encoder e decoder GRU.

    Durante o treino, o decoder comeca com SOS e pode usar teacher forcing.
    """

    def __init__(self, config: GRUSeq2SeqConfig):
        super().__init__()
        self.config = config
        self.encoder = EncoderGRU(VOCAB_SIZE, config.embed_size, config.hidden_size)
        self.decoder = DecoderGRU(VOCAB_SIZE, config.embed_size, config.hidden_size)

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        teacher_forcing_ratio: float,
    ) -> torch.Tensor:
        batch_size = src.size(0)
        tgt_len = tgt.size(1)
        device = src.device

        outputs = torch.zeros(batch_size, tgt_len, VOCAB_SIZE, device=device)

        _, hidden = self.encoder(src)

        input_token = torch.full(
            (batch_size,),
            SOS,
            dtype=torch.long,
            device=device,
        )

        for t in range(tgt_len):
            logits, hidden = self.decoder(input_token, hidden)
            outputs[:, t, :] = logits

            predicted_token = logits.argmax(dim=1)
            use_teacher_forcing = random.random() < teacher_forcing_ratio

            if use_teacher_forcing:
                input_token = tgt[:, t]
            else:
                input_token = predicted_token

        return outputs


@torch.no_grad()
def predict_gru(
    model: GRUSeq2Seq,
    digits: List[int],
    device: torch.device,
    max_len: int | None = None,
    block_invalid_tokens: bool = True,
) -> List[int]:
    """Gera a sequencia invertida com o modelo GRU."""
    model.eval()

    if max_len is None:
        max_len = len(digits) + 3

    src = [int(d) + DIGIT_OFFSET for d in digits] + [EOS]
    src_tensor = torch.tensor([src], dtype=torch.long, device=device)

    _, hidden = model.encoder(src_tensor)
    input_token = torch.tensor([SOS], dtype=torch.long, device=device)
    predicted_tokens: List[int] = []

    for _ in range(max_len):
        logits, hidden = model.decoder(input_token, hidden)

        if block_invalid_tokens:
            logits[:, PAD] = -float("inf")
            logits[:, SOS] = -float("inf")

        next_token = logits.argmax(dim=1)
        token_id = int(next_token.item())

        if token_id == EOS:
            break

        predicted_tokens.append(token_id)
        input_token = next_token

    return decode_tokens_int(predicted_tokens)
