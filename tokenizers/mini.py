"""
Minimal toy implementation of byte-pair encoding.
"""

from typing import DefaultDict
from rich.console import Console
from rich.table import Table
import regex

console = Console()

VOCAB_SIZE = 300

EXAMPLE_STRING = "Hello world! こんにちは世界！🌍 This is a test. これはテストです。"
GPT2_SPLIT_PATTERN = regex.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?[\p{L}]+| ?[\p{N}]+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


def get_pair_counts_from_text(text: str) -> dict[tuple[int, int], int]:
    counts = DefaultDict(int)
    words: list[list[bytes]] = [
        [bytes([b]) for b in word.encode("utf-8")]
        for word in regex.findall(GPT2_SPLIT_PATTERN, text)
    ]
    for word in words:
        pairs = list(zip(word, word[1:]))
        for pair in pairs:
            counts[pair] += 1
    return counts


def get_pair_counts_from_bytes(byte_list: list[int]) -> dict[tuple[int, int], int]:
    counts = DefaultDict(int)
    pairs = list(zip(byte_list, byte_list[1:]))
    for pair in pairs:
        counts[pair] += 1
    # return list(sorted(counts.items(), key=lambda kv: kv[1], reverse=True))
    return counts


def examine_bytes(text: str):
    byte_list = [f"{x:02x}" for x in text.encode("utf-8")]

    table = Table(show_header=False)
    table.add_row("Text", text)
    table.add_row("Length", str(len(text)))
    table.add_row("Bytes", " ".join(byte_list))
    table.add_row("Length", str(len(byte_list)))
    console.print(table)

    pairs_sorted = get_pair_counts_from_text(text)
    console.print(pairs_sorted)


def _get_max_pair(counts: dict[tuple[int, int], int]) -> tuple[int, int]:
    return max(counts.items(), key=lambda kv: kv[1])[0]


def replace_pair_with_token(tokens: list[int], pair: tuple[int, int], new_token: int):
    i = 0
    while i < len(tokens) - 1:
        if (tokens[i], tokens[i + 1]) == pair:
            yield new_token
            i += 2
            continue
        yield tokens[i]
        i += 1
    if i < len(tokens):
        yield tokens[-1]


def encode(text: str) -> tuple[list[int], dict[tuple[int, int], int]]:
    merges = {}
    counts = get_pair_counts_from_text(text)
    tokens = [int(x) for x in text.encode("utf-8")]
    new_token = 256
    pair = _get_max_pair(counts)
    console.log(pair)
    console.log(counts[pair])
    while counts[pair] > 1 and new_token < VOCAB_SIZE:
        merges[pair] = new_token
        tokens = list(replace_pair_with_token(tokens, pair, new_token))
        counts = get_pair_counts_from_bytes(tokens)
        pair = _get_max_pair(counts)
        new_token += 1
    return tokens, merges


def examine_tokenized(text: str, tokens: list[int], merges: dict[tuple[int, int], int]):
    byte_len = len(text.encode("utf-8"))
    table = Table(show_header=False)
    table.add_column()
    table.add_column(justify="right", style="bold")
    table.add_row("Text Len", f"{len(text):,}")
    table.add_row("Byte Len", f"{byte_len:,}")
    table.add_row("Token Len", f"{len(tokens):,}")
    table.add_row("Compression Ratio", f"{byte_len / len(tokens):.2f}x")
    table.add_row("Merge Count", f"{len(merges):,}")
    console.print(table)


def decode(tokens: list[int], merges: dict[tuple[int, int], int]):
    vocab = {i: bytes([i]) for i in range(256)}

    for pair, token in merges.items():
        vocab[token] = vocab[pair[0]] + vocab[pair[1]]

    def gen():
        for token in tokens:
            seq = vocab.get(token, [])
            yield from seq

    return bytes(gen()).decode("utf-8", errors="replace")
