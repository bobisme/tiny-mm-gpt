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
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?[\p{L}]+| ?[\p{N}]+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""",
    regex.IGNORECASE,
)


def str_to_ints(text: str) -> list[list[int]]:
    return [
        [int(b) for b in word.encode("utf-8")]
        for word in regex.findall(GPT2_SPLIT_PATTERN, text)
    ]


def get_pair_counts_from_text(text: str) -> dict[tuple[int, int], int]:
    counts = DefaultDict(int)
    words = str_to_ints(text)
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


class Tokenizer:
    def __init__(self, vocab_size=VOCAB_SIZE) -> None:
        self._built = False
        self.vocab_size = vocab_size
        self.counts: DefaultDict[tuple[int, int], int] = DefaultDict(int)
        self.merges = {}
        self.corpus: list[list[int]] = []

    def add(self, text: str):
        if self._built:
            raise ValueError("Vocabulary already built")
        self.corpus.extend(str_to_ints(text))

    def build(self):
        if self._built:
            raise ValueError("Vocabulary already built")
        self._built = True

        # Count all pairs in the corpus
        for word in self.corpus:
            pairs = list(zip(word, word[1:]))
            for pair in pairs:
                self.counts[pair] += 1

        # Build the vocabulary using BPE
        new_token = 256
        while new_token < self.vocab_size:
            if not self.counts:
                break

            # Find the most frequent pair
            pair = max(self.counts.items(), key=lambda kv: kv[1])[0]
            if self.counts[pair] <= 1:
                break

            # Record the merge
            self.merges[pair] = new_token

            # Update the corpus by replacing the pair with new token
            new_corpus = []
            for word in self.corpus:
                new_word = list(replace_pair_with_token(word, pair, new_token))
                new_corpus.append(new_word)
            self.corpus = new_corpus

            # Recompute counts
            self.counts = DefaultDict(int)
            for word in self.corpus:
                pairs = list(zip(word, word[1:]))
                for pair in pairs:
                    self.counts[pair] += 1

            new_token += 1

    def encode(self, text: str) -> list[int]:
        if not self._built:
            raise ValueError("Vocabulary not built yet. Call build() first.")

        tokens = [int(x) for x in text.encode("utf-8")]

        # Apply merges in order
        for pair, new_token in self.merges.items():
            tokens = list(replace_pair_with_token(tokens, pair, new_token))

        return tokens

    def decode(self, tokens: list[int]) -> str:
        if not self._built:
            raise ValueError("Vocabulary not built yet. Call build() first.")

        # Build vocabulary mapping
        vocab = {i: bytes([i]) for i in range(256)}

        for pair, token in self.merges.items():
            vocab[token] = vocab[pair[0]] + vocab[pair[1]]

        # Decode tokens
        def gen():
            for token in tokens:
                seq = vocab.get(token, [])
                yield from seq

        return bytes(gen()).decode("utf-8", errors="replace")
