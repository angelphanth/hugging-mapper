import os
import shutil
from typing import Callable

import pandas as pd
import pytest
import torch
from transformers import (
    AutoModel,
    AutoTokenizer,
)

from hugger.mapper import (
    HuggingMapper,
    NodeMapper,
    get_tokens,
    map_pooling,
)

# note there are more tests in the function docstrings > Examples


@pytest.fixture
def cache_dir():
    return "pytesting_cache"


@pytest.fixture
def pool_str():
    return "attention_pooling"


@pytest.fixture
def model_name():
    return "sentence-transformers/all-MiniLM-L12-v2"


@pytest.fixture
def text_input():
    return "Sunsets are beautiful."


@pytest.fixture
def tokenizer_kwargs():
    return dict(padding=True, truncation=True, return_tensors="pt", max_length=512)


@pytest.fixture
def tokenizer(model_name, cache_dir):
    yield AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    # clean up
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)


@pytest.fixture
def model(model_name, cache_dir):
    yield AutoModel.from_pretrained(model_name, cache_dir=cache_dir)
    # clean up
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)


@pytest.fixture
def tokens(tokenizer, text_input, tokenizer_kwargs):
    return tokenizer(text_input, **tokenizer_kwargs)


@pytest.fixture
def embedding(model, text_input):
    with torch.no_grad():
        yield model(**text_input)


def test_get_tokens(tokenizer, text_input, tokenizer_kwargs, tokens):
    gen_tokens = get_tokens(tokenizer, text_input, **tokenizer_kwargs)
    assert torch.equal(gen_tokens["input_ids"], tokens["input_ids"])
    assert torch.equal(gen_tokens["attention_mask"], tokens["attention_mask"])
    # clean up
    del gen_tokens


def test_map_pooling(pool_str):
    assert isinstance(map_pooling(pool_str), Callable)


def test_HuggingMapper(model_name, tokenizer_kwargs, pool_str, text_input):
    mapper = HuggingMapper(model_name, pooling=pool_str, **tokenizer_kwargs)
    # first test
    emb = mapper.embed_text(text_input)
    assert emb.shape[0] == 1
    # second test
    emb = mapper.embed_text([text_input] * 3)
    assert emb.shape[0] == 3
    # third test
    assert torch.equal(emb[0], emb[1])
    # clean up
    del mapper, emb


@pytest.fixture
def df():
    return pd.DataFrame(
        {"id": ["n1", "n2", "n3"], "text": ["happy", "doughnut", "foundation"]}
    )


def test_NodeMapper_matches(df, model_name, tokenizer_kwargs, pool_str, text_input):
    mapper = NodeMapper(
        df,
        text_col="text",
        id_col="id",
        model_name=model_name,
        pooling=pool_str,
        **tokenizer_kwargs,
    )
    # first test
    emb = mapper.embed_text([text_input] * 3)
    assert emb.shape[0] == 3

    # second test, get_match
    match, meta = mapper.get_match("concrete")
    assert match == "n3"
    assert meta["text"] == "foundation"
    assert meta["score"] > 0.4

    # third test, get_match, score close to 1
    match, meta = mapper.get_match("happy", threshold=0.8)
    assert match == "n1"
    assert meta["text"] == "happy"
    assert meta["score"] > 0.99

    # clean up
    del mapper, emb, match, meta


def test_NodeMapper_similar(df, model_name, tokenizer_kwargs, pool_str):
    mapper = NodeMapper(
        df,
        text_col="text",
        id_col="id",
        model_name=model_name,
        pooling=pool_str,
        **tokenizer_kwargs,
    )
    # first test
    matches = mapper.get_similar("concrete", threshold=0.3)
    assert "n3" in matches.keys()

    # top_k all
    matches = mapper.get_similar("cheerful", threshold=0, top_k=None)
    assert len(matches) == 3
    assert list(matches.keys())[0] == "n1"
    assert matches["n1"]["score"] > 0.5
    assert matches == mapper.get_similar("cheerful", threshold=0, top_k=1000)

    # top_k 2
    matches = mapper.get_similar("straw", threshold=0, top_k=2)
    assert len(matches) == 2
    # assert score under 0.3
    for i in matches:
        assert matches[i]["score"] < 0.3

    # clean up
    del mapper, matches
