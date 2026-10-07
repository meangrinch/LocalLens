from model_utils import fit_batch_size

MIB = 2**20


def test_fit_batch_size_caps_at_default_and_vram():
    # so400m-512 measured at ~26 MiB per image; 70% of free VRAM is the budget.
    assert fit_batch_size(4800 * MIB, 26 * MIB, max_batch_size=16) == 16
    assert (
        fit_batch_size(500 * MIB, 26 * MIB, max_batch_size=16) == 12
    )  # 13.5 -> step of 4
    assert fit_batch_size(50 * MIB, 26 * MIB, max_batch_size=16) == 4  # slider minimum
