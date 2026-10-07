import pytest

from app import (
    AVAILABLE_MODELS,
    MODEL_BATCH_SIZE_DEFAULTS,
    MODEL_CONFIDENCE_DEFAULTS,
    generate_db_path_for_model,
    get_simplified_model_identifier,
)
from model_utils import get_model_type


@pytest.mark.parametrize("model", AVAILABLE_MODELS)
def test_confidence_defaults(model):
    conf = MODEL_CONFIDENCE_DEFAULTS[model]
    thresh = "siglip_thresh" if get_model_type(model) == "siglip" else "clip_thresh"
    assert {thresh, "combined_thresh", "image_only_thresh"} <= conf.keys()


@pytest.mark.parametrize("model", AVAILABLE_MODELS)
def test_batch_size_defaults(model):
    bs = MODEL_BATCH_SIZE_DEFAULTS[model]
    assert isinstance(bs, int)
    assert 4 <= bs <= 256


def test_model_identifiers_are_unique():
    identifiers = [get_simplified_model_identifier(m) for m in AVAILABLE_MODELS]
    assert len(identifiers) == len(set(identifiers))


def test_database_paths_are_unique():
    db_paths = [generate_db_path_for_model(m) for m in AVAILABLE_MODELS]
    assert len(db_paths) == len(set(db_paths))


@pytest.mark.parametrize(
    "model",
    [
        "google/siglip2-base-patch16-224",
        "google/siglip2-so400m-patch14-384",
        "google/siglip2-so400m-patch16-512",
        "google/siglip2-giant-opt-patch16-384",
    ],
)
def test_siglip2_model_type(model):
    assert get_model_type(model) == "siglip"
