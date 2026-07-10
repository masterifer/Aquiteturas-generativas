"""
Utilitarios de dados para o experimento de inversao de sequencias.

A tarefa do experimento e simples:
entrada: [1, 5, 9, 2]
saida:   [2, 9, 5, 1]

Internamente o modelo nao usa os digitos puros. Ele usa IDs de tokens.
"""

from __future__ import annotations

import random
from typing import Iterable, List, Sequence, Tuple

import torch


PAD = 0
SOS = 1
EOS = 2
DIGIT_OFFSET = 3
VOCAB_SIZE = 13


def tokenize_digits(digits: Sequence[int]) -> List[int]:
    """Converte digitos 0-9 para IDs de tokens."""
    return [int(d) + DIGIT_OFFSET for d in digits]


def detokenize_token(token: int) -> int:
    """Converte um token de digito de volta para o digito original."""
    return int(token) - DIGIT_OFFSET


def pad_sequence(seq: Sequence[int], max_len: int) -> List[int]:
    """Preenche uma sequencia com PAD ate atingir max_len."""
    return list(seq) + [PAD] * (max_len - len(seq))


def generate_digits(min_len: int, max_len: int) -> List[int]:
    """Gera uma sequencia aleatoria de digitos."""
    length = random.randint(min_len, max_len)
    return [random.randint(0, 9) for _ in range(length)]


def generate_gru_example(min_len: int, max_len: int) -> Tuple[List[int], List[int]]:
    """
    Gera um exemplo para o Seq2Seq com GRU.

    Retorna:
    src: entrada do encoder, com EOS no final.
    tgt: alvo do decoder, com EOS no final.

    Exemplo:
    digits = [1, 5, 9, 2]
    src = [4, 8, 12, 5, 2]
    tgt = [5, 12, 8, 4, 2]
    """
    digits = generate_digits(min_len, max_len)
    src = tokenize_digits(digits) + [EOS]
    tgt = tokenize_digits(list(reversed(digits))) + [EOS]
    return src, tgt


def generate_transformer_example(min_len: int, max_len: int) -> Tuple[List[int], List[int], List[int]]:
    """
    Gera um exemplo para o Transformer encoder-decoder.

    Retorna:
    src: entrada do encoder.
    decoder_input: entrada deslocada do decoder, com SOS no inicio.
    decoder_target: alvo do decoder, com EOS no final.

    Exemplo:
    digits = [1, 5, 9, 2]
    src            = [1, 5, 9, 2, EOS]
    decoder_input  = [SOS, 2, 9, 5, 1]
    decoder_target = [2, 9, 5, 1, EOS]
    """
    digits = generate_digits(min_len, max_len)
    reversed_tokens = tokenize_digits(list(reversed(digits)))

    src = tokenize_digits(digits) + [EOS]
    decoder_input = [SOS] + reversed_tokens
    decoder_target = reversed_tokens + [EOS]

    return src, decoder_input, decoder_target


def generate_gru_batch(
    batch_size: int,
    min_len: int,
    max_len: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Gera um batch para o Seq2Seq com GRU."""
    examples = [generate_gru_example(min_len, max_len) for _ in range(batch_size)]

    src_seqs = [ex[0] for ex in examples]
    tgt_seqs = [ex[1] for ex in examples]

    max_src_len = max(len(seq) for seq in src_seqs)
    max_tgt_len = max(len(seq) for seq in tgt_seqs)

    src_batch = [pad_sequence(seq, max_src_len) for seq in src_seqs]
    tgt_batch = [pad_sequence(seq, max_tgt_len) for seq in tgt_seqs]

    src_tensor = torch.tensor(src_batch, dtype=torch.long, device=device)
    tgt_tensor = torch.tensor(tgt_batch, dtype=torch.long, device=device)

    return src_tensor, tgt_tensor


def generate_transformer_batch(
    batch_size: int,
    min_len: int,
    max_len: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Gera um batch para o Transformer encoder-decoder."""
    examples = [generate_transformer_example(min_len, max_len) for _ in range(batch_size)]

    src_seqs = [ex[0] for ex in examples]
    dec_in_seqs = [ex[1] for ex in examples]
    dec_tgt_seqs = [ex[2] for ex in examples]

    max_src_len = max(len(seq) for seq in src_seqs)
    max_dec_len = max(len(seq) for seq in dec_tgt_seqs)

    src_batch = [pad_sequence(seq, max_src_len) for seq in src_seqs]
    dec_in_batch = [pad_sequence(seq, max_dec_len) for seq in dec_in_seqs]
    dec_tgt_batch = [pad_sequence(seq, max_dec_len) for seq in dec_tgt_seqs]

    src_tensor = torch.tensor(src_batch, dtype=torch.long, device=device)
    dec_in_tensor = torch.tensor(dec_in_batch, dtype=torch.long, device=device)
    dec_tgt_tensor = torch.tensor(dec_tgt_batch, dtype=torch.long, device=device)

    return src_tensor, dec_in_tensor, dec_tgt_tensor


def decode_tokens_int(tokens: Iterable[int]) -> List[int]:
    """Converte uma sequencia de tokens em digitos inteiros."""
    result: List[int] = []

    for token in tokens:
        if torch.is_tensor(token):
            token = int(token.item())
        else:
            token = int(token)

        if token == EOS:
            break
        if token in (PAD, SOS):
            continue

        result.append(detokenize_token(token))

    return result


def parse_test_sequences(raw_text: str) -> List[List[int]]:
    """
    Converte o texto do textarea da interface em uma lista de sequencias.

    Aceita linhas como:
    3,3,1,9,0,5,6
    [3, 1, 1, 4]
    """
    tests: List[List[int]] = []

    for line in raw_text.splitlines():
        line = line.strip()
        if not line:
            continue

        clean = line.replace("[", "").replace("]", "").replace(";", ",")
        values = [part.strip() for part in clean.split(",") if part.strip()]
        sequence = [int(value) for value in values]

        if not sequence:
            continue
        if any(value < 0 or value > 9 for value in sequence):
            raise ValueError("Todas as sequencias devem conter apenas digitos de 0 a 9.")

        tests.append(sequence)

    if not tests:
        raise ValueError("Informe pelo menos uma sequencia de teste.")

    return tests
