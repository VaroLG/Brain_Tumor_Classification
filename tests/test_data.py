import numpy as np
import pandas as pd
from PIL import Image

from btc.data.folders import build_folder_manifest, dhash
from btc.data.labels import build_categorical_labels, build_survival_labels, survival_label
from btc.data.preprocessing import crop_to_foreground, normalize_to_uint8


# --- preprocesado ------------------------------------------------------------
def test_crop_removes_black_frame():
    img = np.zeros((100, 120), np.uint8)
    img[30:60, 40:90] = 200
    out = crop_to_foreground(img, margin=0)
    assert out.shape == (30, 50)


def test_crop_rgb_and_empty_image():
    rgb = np.zeros((50, 50, 3), np.uint8)
    rgb[10:20, 10:30] = 255
    assert crop_to_foreground(rgb, margin=0).shape == (10, 20, 3)
    empty = np.zeros((50, 50), np.uint8)
    assert crop_to_foreground(empty).shape == (50, 50)


def test_normalize_constant_image_does_not_produce_nan():
    out = normalize_to_uint8(np.full((10, 10), 7.0))
    assert out.dtype == np.uint8 and not np.isnan(out.astype(float)).any()


def test_normalize_is_robust_to_outlier():
    img = np.linspace(1, 100, 10_000).reshape(100, 100)
    img[0, 0] = 1e6  # un vóxel extremo
    out = normalize_to_uint8(img)
    # con min-max puro casi todo quedaría en 0; con percentiles se usa todo el rango
    assert np.median(out) > 100


# --- etiquetas clínicas ---------------------------------------------------------
def test_survival_label_handles_censoring():
    assert survival_label(100, event=1, threshold_days=365) == "short"  # murió pronto
    assert survival_label(500, event=1, threshold_days=365) == "long"
    assert survival_label(500, event=0, threshold_days=365) == "long"  # vivo y ya pasó T
    assert survival_label(100, event=0, threshold_days=365) is None  # censurado: desconocido


def test_build_survival_labels_uses_median_and_drops_censored():
    clinical = pd.DataFrame(
        {
            "ID": ["A", "B", "C", "D", "E"],
            "days": [100, 200, 300, 400, "Not Available"],
            "dead": [1, 0, 1, 1, 1],
        }
    )
    out = build_survival_labels(clinical, "ID", "days", "dead")
    assert out.attrs["threshold_days"] == 250
    assert dict(zip(out.patient_id, out.label, strict=True)) == {
        "A": "short",
        "C": "long",
        "D": "long",
    }  # B censurado antes de T, E sin dato


def test_categorical_labels_whitelist():
    mgmt = ["Methylated", "Unmethylated", "Not Available"]
    clinical = pd.DataFrame({"ID": [1, 2, 3], "MGMT": mgmt})
    out = build_categorical_labels(
        clinical, "ID", "MGMT", {"Methylated": "methylated", "Unmethylated": "unmethylated"}
    )
    assert list(out.label) == ["methylated", "unmethylated"]


# --- carpetas + dHash -------------------------------------------------------
def test_folder_manifest_groups_near_duplicates(tmp_path):
    rng = np.random.default_rng(0)
    a = (rng.random((64, 64)) * 255).astype(np.uint8)
    b = (rng.random((64, 64)) * 255).astype(np.uint8)
    (tmp_path / "Training" / "glioma").mkdir(parents=True)
    (tmp_path / "Testing" / "glioma").mkdir(parents=True)
    Image.fromarray(a).save(tmp_path / "Training" / "glioma" / "1.png")
    # mismo contenido reescalado y en JPEG en "Testing": casi-duplicado
    dup = tmp_path / "Testing" / "glioma" / "1b.jpg"
    Image.fromarray(a).resize((128, 128)).save(dup, quality=85)
    Image.fromarray(b).save(tmp_path / "Training" / "glioma" / "2.png")

    df = build_folder_manifest(tmp_path)
    assert len(df) == 3
    assert df["patient_id"].nunique() == 2  # el duplicado comparte grupo
    assert set(df["label"]) == {"glioma"}
    assert len(dhash(tmp_path / "Training" / "glioma" / "1.png")) == 16
