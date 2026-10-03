"""Entrenamiento en dos fases con transfer learning.

Fase 1 — "feature extraction": la base preentrenada está congelada y solo se
entrena la cabeza nueva con una LR relativamente alta. Es lo que hacía el TFM.

Fase 2 — "fine-tuning": se descongelan las últimas capas de la base y se sigue
entrenando con una LR 100 veces menor, para adaptar los filtros de alto nivel
(texturas, formas) de "objetos de ImageNet" a "tejido cerebral en RM". El TFM
lo dejaba como trabajo futuro; suele ser la mejora más grande en imagen médica.

Ambas fases usan:
- EarlyStopping sobre ``val_loss`` restaurando los mejores pesos. Sustituye a
  elegir el nº de épocas a mano (en el TFM, 60 épocas daban peor validación
  que 15: se estaba sobreajustando sin detenerse).
- ``class_weight`` para que las clases minoritarias pesen lo mismo en la loss.
- El test se usa UNA sola vez, al final. Toda decisión (épocas, conservar o no
  el fine-tuning) se toma con validación.
"""

from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path

import keras
import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight

from btc.config import ExperimentConfig
from btc.datasets import make_dataset
from btc.evaluate import evaluate_predictions
from btc.models import build_model, unfreeze_top_layers
from btc.splits import assert_no_patient_leakage, split_summary
from btc.utils import get_logger, set_seed

log = get_logger(__name__)


def _callbacks(run_dir: Path, patience: int, phase: str) -> list[keras.callbacks.Callback]:
    return [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=patience, restore_best_weights=True, verbose=1
        ),
        keras.callbacks.CSVLogger(run_dir / f"history_{phase}.csv"),
    ]


def _class_weights(labels: pd.Series, classes: list[str]) -> dict[int, float]:
    """Pesos inversamente proporcionales a la frecuencia de cada clase en train."""
    y = labels.map({c: i for i, c in enumerate(classes)}).to_numpy()
    w = compute_class_weight("balanced", classes=np.arange(len(classes)), y=y)
    return {i: float(v) for i, v in enumerate(w)}


def train(cfg: ExperimentConfig, run_dir: str | Path | None = None) -> Path:
    """Entrena y evalúa un experimento completo. Devuelve la carpeta del run."""
    set_seed(cfg.train.seed)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(run_dir or Path(cfg.output_dir) / f"{cfg.name}_{stamp}")
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg.to_yaml(run_dir / "config.yaml")

    # --- Datos -------------------------------------------------------------
    df = pd.read_csv(cfg.data.manifest)
    if "split" not in df.columns:
        raise ValueError("El manifest no tiene columna 'split'. Ejecuta antes `btc split`.")
    assert_no_patient_leakage(df)  # salvaguarda: nunca entrenar con fuga
    shutil.copy(cfg.data.manifest, run_dir / "manifest.csv")
    log.info("Resumen del split:\n%s", split_summary(df))

    parts = {s: df[df["split"] == s].reset_index(drop=True) for s in ("train", "val", "test")}
    common = dict(
        classes=cfg.data.classes,
        backbone=cfg.model.backbone,
        image_size=cfg.data.image_size,
        batch_size=cfg.data.batch_size,
        seed=cfg.train.seed,
    )
    ds_train = make_dataset(parts["train"], training=True, augment=cfg.train.augment, **common)
    ds_val = make_dataset(parts["val"], **common)
    ds_test = make_dataset(parts["test"], **common)
    cw = None
    if cfg.train.class_weights:
        cw = _class_weights(parts["train"]["label"], cfg.data.classes)
    log.info("Pesos de clase: %s", cw)

    # --- Fase 1: cabeza -----------------------------------------------------
    model, base = build_model(
        cfg.model.backbone,
        len(cfg.data.classes),
        cfg.data.image_size,
        cfg.model.dense_units,
        cfg.model.dropout,
        cfg.model.weights,
    )
    model.compile(
        optimizer=keras.optimizers.Adam(cfg.train.head_lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    log.info("Fase 1: %d épocas máx., LR=%g", cfg.train.head_epochs, cfg.train.head_lr)
    h1 = model.fit(
        ds_train,
        validation_data=ds_val,
        epochs=cfg.train.head_epochs,
        class_weight=cw,
        verbose=2,
        callbacks=_callbacks(run_dir, cfg.train.early_stopping_patience, "head"),
    )
    best_head = min(h1.history["val_loss"])
    # EarlyStopping ya restauró los mejores pesos de la fase 1
    model.save(run_dir / "model.keras")

    # --- Fase 2: fine-tuning --------------------------------------------------
    pretrained = cfg.model.weights is not None
    do_finetune = pretrained and cfg.train.finetune_layers > 0 and cfg.train.finetune_epochs > 0
    if not pretrained:
        log.info("Sin pesos preentrenados: la red se entrenó entera en la fase 1, no hay fase 2")
    if do_finetune:
        head_weights = run_dir / "phase1.weights.h5"
        model.save_weights(head_weights)
        n = unfreeze_top_layers(base, cfg.train.finetune_layers)
        # Recompilar es obligatorio para que el cambio de 'trainable' surta efecto
        model.compile(
            optimizer=keras.optimizers.Adam(cfg.train.finetune_lr),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        log.info("Fase 2: %d capas entrenables, LR=%g", n, cfg.train.finetune_lr)
        h2 = model.fit(
            ds_train,
            validation_data=ds_val,
            epochs=cfg.train.finetune_epochs,
            class_weight=cw,
            verbose=2,
            callbacks=_callbacks(run_dir, cfg.train.early_stopping_patience, "finetune"),
        )
        if min(h2.history["val_loss"]) < best_head:
            log.info("El fine-tuning mejora la validación: se conserva")
        else:
            # Si descongelar no ayuda, volvemos a los pesos de la fase 1 en vez de
            # quedarnos con un modelo peor (decisión tomada SOLO con validación)
            log.info("El fine-tuning no mejora val_loss: se restauran los pesos de la fase 1")
            model.load_weights(head_weights)
        model.save(run_dir / "model.keras")
        head_weights.unlink()

    # --- Evaluación en test (una sola vez, al final) ---------------------------
    probs = model.predict(ds_test, verbose=0)
    y_true = parts["test"]["label"].map({c: i for i, c in enumerate(cfg.data.classes)}).to_numpy()
    report = evaluate_predictions(
        parts["test"]["patient_id"].to_numpy(),
        y_true,
        probs,
        cfg.data.classes,
        run_dir,
        seed=cfg.train.seed,
    )
    pl, ci = report["patient_level"], report["patient_level_ci95"]
    log.info(
        "Test (paciente, n=%d): bal_acc=%.3f IC95 %s | AUC=%s",
        report["n_patients"],
        pl["balanced_accuracy"],
        ci.get("balanced_accuracy"),
        pl.get("auc"),
    )
    log.info("Resultados en %s", run_dir)
    return run_dir
