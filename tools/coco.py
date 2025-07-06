from pathlib import Path
from typing import TypedDict
from datasets import load_dataset

from rich import print

from tokenizers.mini import Tokenizer, save_tokenizer


class Sample(TypedDict):
    captions: list[str]


def get_captions():
    dataset = load_dataset("phiyodr/coco2017", split="train[:1%]")
    for item in dataset:
        yield from (x.strip() for x in item["captions"])  # type: ignore


def build_tokenizer_from_captions(
    save_to: Path, vocab_size: int | None = None
) -> Tokenizer:
    tokenizer = Tokenizer(vocab_size=vocab_size)
    for cap in get_captions():
        tokenizer.add(cap)
    tokenizer.build()
    save_tokenizer(tokenizer, save_to)
    print(f"saved to {save_to}")

    return tokenizer
