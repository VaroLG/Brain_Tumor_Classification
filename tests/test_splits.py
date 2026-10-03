import numpy as np
import pandas as pd
import pytest

from btc.splits import assert_no_patient_leakage, patient_level_split


@pytest.fixture
def manifest() -> pd.DataFrame:
    """60 pacientes, 2 clases desbalanceadas (2:1) y nº de cortes variable."""
    rng = np.random.default_rng(0)
    rows = []
    for p in range(60):
        label = "a" if p < 40 else "b"
        for s in range(rng.integers(1, 12)):
            rows.append({"image_path": f"{p}_{s}.png", "patient_id": f"P{p}", "label": label})
    return pd.DataFrame(rows)


def test_no_patient_in_two_splits(manifest):
    out = patient_level_split(manifest, 0.15, 0.2, seed=1)
    assert out.groupby("patient_id")["split"].nunique().max() == 1


def test_every_row_gets_a_split(manifest):
    out = patient_level_split(manifest, seed=1)
    assert out["split"].notna().all()
    assert set(out["split"]) == {"train", "val", "test"}


def test_split_is_reproducible(manifest):
    a = patient_level_split(manifest, seed=7)["split"]
    b = patient_level_split(manifest, seed=7)["split"]
    pd.testing.assert_series_equal(a, b)


def test_both_classes_in_every_split(manifest):
    out = patient_level_split(manifest, seed=3)
    assert (out.groupby("split")["label"].nunique() == 2).all()


def test_patient_fractions_are_approximate(manifest):
    out = patient_level_split(manifest, 0.15, 0.2, seed=3)
    frac = out.drop_duplicates("patient_id")["split"].value_counts(normalize=True)
    assert abs(frac["test"] - 0.2) < 0.1
    assert abs(frac["val"] - 0.15) < 0.1


def test_leakage_is_detected():
    leaky = pd.DataFrame({"patient_id": ["P1", "P1", "P2"], "split": ["train", "test", "val"]})
    with pytest.raises(ValueError, match="Fuga de datos"):
        assert_no_patient_leakage(leaky)


def test_invalid_fractions(manifest):
    with pytest.raises(ValueError):
        patient_level_split(manifest, 0.6, 0.5)
