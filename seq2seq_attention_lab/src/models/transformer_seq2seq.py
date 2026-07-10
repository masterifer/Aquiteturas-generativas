"""Modelo Transformer encoder-decoder para inversao de sequencias."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn

from src.data import DIGIT_OFFSET, EOS, PAD, SOS, VOCAB_SIZE, decode_tokens_int


@dataclass
class TransformerSeq2SeqConfig:
    d_model: int = 16
    nhead: int = 2
    dim_feedforward: int = 64
    num_encoder_layers: int = 2
    num_decoder_layers: int = 2
    dropout: float = 0.1
    max_seq_len: int = 64


class TransformerSeq2Seq(nn.Module):
    """
    Transformer encoder-decoder.

    O encoder usa self-attention para representar a entrada.
    O decoder usa self-attention causal para a saida parcial e cross-attention
    para consultar as representacoes do encoder.
    """

    def __init__(self, config: TransformerSeq2SeqConfig):
        super().__init__()

        if config.d_model % config.nhead != 0:
            raise ValueError("d_model precisa ser divisivel por nhead.")

        self.config = config
        self.d_model = config.d_model

        self.token_embedding = nn.Embedding(VOCAB_SIZE, config.d_model, padding_idx=PAD)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.d_model)

        self.transformer = nn.Transformer(
            d_model=config.d_model,
            nhead=config.nhead,
            num_encoder_layers=config.num_encoder_layers,
            num_decoder_layers=config.num_decoder_layers,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True,
        )

        self.output_layer = nn.Linear(config.d_model, VOCAB_SIZE)

    def add_position_embedding(self, tokens: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = tokens.shape

        if seq_len > self.config.max_seq_len:
            raise ValueError(
                f"Sequencia com tamanho {seq_len} excede max_seq_len={self.config.max_seq_len}."
            )

        positions = torch.arange(seq_len, device=tokens.device).unsqueeze(0).expand(batch_size, seq_len)
        token_emb = self.token_embedding(tokens)
        pos_emb = self.position_embedding(positions)

        return token_emb * math.sqrt(self.d_model) + pos_emb

    def make_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()

    def forward(self, src: torch.Tensor, decoder_input: torch.Tensor) -> torch.Tensor:
        src_emb = self.add_position_embedding(src)
        tgt_emb = self.add_position_embedding(decoder_input)

        src_key_padding_mask = src == PAD
        tgt_key_padding_mask = decoder_input == PAD
        tgt_mask = self.make_causal_mask(decoder_input.size(1), decoder_input.device)

        transformer_output = self.transformer(
            src=src_emb,
            tgt=tgt_emb,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )

        return self.output_layer(transformer_output)


@torch.no_grad()
def predict_transformer(
    model: TransformerSeq2Seq,
    digits: List[int],
    device: torch.device,
    max_len: int | None = None,
    block_invalid_tokens: bool = True,
) -> List[int]:
    """Gera a sequencia invertida com o Transformer."""
    model.eval()

    if max_len is None:
        max_len = len(digits) + 3

    src = [int(d) + DIGIT_OFFSET for d in digits] + [EOS]
    src_tensor = torch.tensor([src], dtype=torch.long, device=device)

    generated_tokens = [SOS]

    for _ in range(max_len):
        decoder_input = torch.tensor([generated_tokens], dtype=torch.long, device=device)
        outputs = model(src_tensor, decoder_input)
        next_token_logits = outputs[:, -1, :]

        if block_invalid_tokens:
            next_token_logits[:, PAD] = -float("inf")
            next_token_logits[:, SOS] = -float("inf")

        next_token = int(next_token_logits.argmax(dim=1).item())

        if next_token == EOS:
            break

        generated_tokens.append(next_token)

    predicted_tokens = generated_tokens[1:]
    return decode_tokens_int(predicted_tokens)
