import pytest
import torch

from kestrel.model import GPT, ModelConfig

SENTENCE = "to be, or not to be, that is the question:\n"
TEXT = SENTENCE * 300 + "whether 'tis nobler in the mind to suffer\n" * 60


def tiny_cfg(**kw) -> ModelConfig:
    base = dict(vocab_size=32, block_size=32, n_layer=2, n_head=2, d_model=32, dropout=0.0)
    base.update(kw)
    return ModelConfig(**base)


def make_model(seed: int = 0, **kw) -> GPT:
    torch.manual_seed(seed)
    return GPT(tiny_cfg(**kw))


@pytest.fixture
def text_file(tmp_path):
    p = tmp_path / "corpus.txt"
    p.write_text(TEXT, encoding="utf-8")
    return p
