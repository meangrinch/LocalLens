import os

from app import DB_ROOT, generate_db_path_for_model


def test_each_soft_token_value_gets_its_own_index_and_280_keeps_the_original():
    gemma = "google/embeddinggemma-2"
    original = os.path.join(DB_ROOT, "embeddinggemma_2")
    assert generate_db_path_for_model(gemma) == original
    assert generate_db_path_for_model(gemma, 280) == original
    assert generate_db_path_for_model(gemma, 140) == original + "_140"
    assert generate_db_path_for_model(gemma, 1120) == original + "_1120"
    assert generate_db_path_for_model(
        "google/siglip2-base-patch16-224"
    ) == os.path.join(DB_ROOT, "siglip2-base")
