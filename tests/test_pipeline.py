"""Tests de evaluación y una ejecución de punta a punta con datos sintéticos.

La prueba end-to-end usa pesos aleatorios (``weights=None``) e imágenes de
64x64 para que corra en CPU en segundos: comprueba que el código funciona,
no que el modelo sea bueno.
"""

import json

import numpy as np
import pytest

from btc.evaluate import aggregate_by_patient, bootstrap_ci, compute_metrics


def test_aggregate_by_patient_averages_probabilities():
    pids = np.array(["A", "A", "B"])
    y = np.array([0, 0, 1])
    probs = np.array([[0.9, 0.1], [0.5, 0.5], [0.2, 0.8]])
    ids, yp, pp = aggregate_by_patient(pids, y, probs)
    assert list(ids) == ["A", "B"]
    np.testing.assert_allclose(pp[0], [0.7, 0.3])
    assert list(yp) == [0, 1]


def test_aggregate_rejects_inconsistent_labels():
    with pytest.raises(ValueError):
        aggregate_by_patient(np.array(["A", "A"]), np.array([0, 1]), np.eye(2))


def test_metrics_perfect_and_ci():
    y = np.array([0, 1] * 20)
    probs = np.eye(2)[y] * 0.8 + 0.1
    m = compute_metrics(y, probs)
    assert m["accuracy"] == 1.0 and m["auc"] == 1.0
    ci = bootstrap_ci(y, probs, n_boot=50)
    assert ci["balanced_accuracy"] == [1.0, 1.0]


def test_balanced_accuracy_exposes_majority_classifier():
    y = np.array([0] * 30 + [1] * 10)
    probs = np.tile([0.9, 0.1], (40, 1))  # siempre predice la clase mayoritaria
    m = compute_metrics(y, probs)
    assert m["accuracy"] == 0.75
    assert m["balanced_accuracy"] == 0.5


@pytest.mark.parametrize("backbone", ["efficientnetb0", "vgg19"])
def test_dataset_shapes_and_no_aug_in_eval(tmp_path, backbone):
    from btc.datasets import make_dataset
    from btc.synthetic import make_synthetic_manifest

    df = make_synthetic_manifest(tmp_path, patients_per_class=2, slices_per_patient=2, size=64)
    ds = make_dataset(df, ["small", "large"], backbone, image_size=64, batch_size=4)
    x, y = next(iter(ds))
    assert x.shape == (4, 64, 64, 3) and y.shape == (4,)
    # Sin aumento, dos pasadas por el dataset de evaluación dan lo mismo
    x2, _ = next(iter(ds))
    np.testing.assert_array_equal(x.numpy(), x2.numpy())


def test_end_to_end_training(tmp_path):
    from btc.config import DataConfig, ExperimentConfig, ModelConfig, TrainConfig
    from btc.splits import patient_level_split
    from btc.synthetic import make_synthetic_manifest
    from btc.train import train

    df = make_synthetic_manifest(
        tmp_path / "img", patients_per_class=10, slices_per_patient=3, size=64
    )
    patient_level_split(df, 0.2, 0.2, seed=0).to_csv(tmp_path / "m.csv", index=False)
    cfg = ExperimentConfig(
        name="smoke",
        data=DataConfig(
            manifest=str(tmp_path / "m.csv"),
            classes=["small", "large"],
            image_size=64,
            batch_size=8,
        ),
        model=ModelConfig(backbone="efficientnetb0", dense_units=[16], weights=None),
        train=TrainConfig(
            head_epochs=1, finetune_epochs=1, finetune_layers=5, early_stopping_patience=1
        ),
    )
    run = train(cfg, run_dir=tmp_path / "run")
    for f in [
        "config.yaml",
        "manifest.csv",
        "model.keras",
        "metrics_test.json",
        "predictions_patient.csv",
        "paciente_confusion.png",
        "history_head.csv",
    ]:
        assert (run / f).exists(), f
    report = json.loads((run / "metrics_test.json").read_text())
    assert report["n_patients"] == report["n_slices"] // 3
    assert not (run / "phase1.weights.h5").exists()
