from typing import TypedDict
from datasets import load_dataset


class Sample(TypedDict):
    captions: list[str]


def get_captions():
    dataset = load_dataset("phiyodr/coco2017", split="train[:1%]")
    for item in dataset:
        yield from (x.strip() for x in item["captions"])  # type: ignore
