"""Evaluación a nivel de corte y a nivel de paciente, con intervalos de confianza.

Por qué evaluar por paciente
----------------------------
Clínicamente la decisión se toma por paciente, no por corte. Además, si un
paciente aporta 10 cortes y otro 2, la métrica por corte pondera 5 veces más al
primero. Aquí se calcula la probabilidad media de los cortes de cada paciente
y se reportan ambas métricas; la de paciente es la que debe citarse.

Por qué intervalos de confianza
-------------------------------
Con ~100 pacientes en test, un 0.72 de exactitud puede ser 0.63–0.81. Sin el
intervalo es imposible saber si EfficientNet "gana" a VGG19 o si la diferencia
es ruido. Se usa bootstrap remuestreando *pacientes* (no cortes).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # sin ventana: las figuras se guardan en disco
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


def compute_metrics(y_true: np.ndarray, probs: np.ndarray) -> dict[str, float]:
    """Métricas principales a partir de etiquetas verdaderas y probabilidades softmax.

    - accuracy: proporción de aciertos (engañosa si las clases están desbalanceadas)
    - balanced_accuracy: media del recall por clase (la clave con desbalance:
      el TFM tenía 2:1 SG baja/alta, y "predecir siempre baja" ya da 0.67 de accuracy)
    - f1_macro: media no ponderada del F1 de cada clase
    - auc: binario -> AUC de la clase 1; multiclase -> one-vs-rest macro
    """
    y_pred = probs.argmax(axis=1)
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }
    present = np.unique(y_true)
    if len(present) > 1:
        if probs.shape[1] == 2:
            out["auc"] = roc_auc_score(y_true, probs[:, 1])
        elif len(present) == probs.shape[1]:
            out["auc"] = roc_auc_score(y_true, probs, multi_class="ovr", average="macro")
    return {k: float(v) for k, v in out.items()}


def aggregate_by_patient(
    patient_ids: np.ndarray, y_true: np.ndarray, probs: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Media de probabilidades por paciente. Devuelve (ids, y_true, probs) por paciente."""
    df = pd.DataFrame(probs).assign(pid=patient_ids, y=y_true)
    grouped = df.groupby("pid", sort=True)
    labels = grouped["y"].agg(lambda s: s.iloc[0])
    if (grouped["y"].nunique() > 1).any():
        raise ValueError("Un paciente tiene cortes con etiquetas distintas")
    p = grouped[list(range(probs.shape[1]))].mean()
    return p.index.to_numpy(), labels.to_numpy(), p.to_numpy()


def bootstrap_ci(
    y_true: np.ndarray, probs: np.ndarray, n_boot: int = 1000, alpha: float = 0.05, seed: int = 42
) -> dict[str, list[float]]:
    """IC (1-alpha) por bootstrap percentil. Recibe datos YA agregados por paciente."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    samples: dict[str, list[float]] = {}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:  # remuestreo degenerado: se descarta
            continue
        for k, v in compute_metrics(y_true[idx], probs[idx]).items():
            samples.setdefault(k, []).append(v)
    return {
        k: [float(np.quantile(v, alpha / 2)), float(np.quantile(v, 1 - alpha / 2))]
        for k, v in samples.items()
    }


def save_figures(
    y_true: np.ndarray, probs: np.ndarray, classes: list[str], out_dir: Path, prefix: str
) -> None:
    """Matriz de confusión (con nombres de clase reales, no "Clase 0") y curvas ROC."""
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ConfusionMatrixDisplay(
        confusion_matrix(y_true, probs.argmax(1), labels=range(len(classes))),
        display_labels=classes,
    ).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Verdadero")
    ax.set_title(f"Matriz de confusión ({prefix})")
    fig.tight_layout()
    fig.savefig(out_dir / f"{prefix}_confusion.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4.5))
    for i, c in enumerate(classes):
        if len(np.unique(y_true == i)) == 2:
            RocCurveDisplay.from_predictions(y_true == i, probs[:, i], name=c, ax=ax)
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1)
    ax.set_title(f"ROC one-vs-rest ({prefix})")
    fig.tight_layout()
    fig.savefig(out_dir / f"{prefix}_roc.png", dpi=150)
    plt.close(fig)


def evaluate_predictions(
    patient_ids: np.ndarray,
    y_true: np.ndarray,
    probs: np.ndarray,
    classes: list[str],
    out_dir: str | Path,
    n_boot: int = 1000,
    seed: int = 42,
) -> dict:
    """Calcula y guarda métricas por corte y por paciente (+IC) y las figuras."""
    out_dir = Path(out_dir)
    pid, yp, pp = aggregate_by_patient(patient_ids, y_true, probs)
    report = {
        "n_slices": int(len(y_true)),
        "n_patients": int(len(pid)),
        "slice_level": compute_metrics(y_true, probs),
        "patient_level": compute_metrics(yp, pp),
        "patient_level_ci95": bootstrap_ci(yp, pp, n_boot=n_boot, seed=seed),
        "classification_report_patient": classification_report(
            yp,
            pp.argmax(1),
            labels=list(range(len(classes))),
            target_names=classes,
            zero_division=0,
            output_dict=True,
        ),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics_test.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    pd.DataFrame(pp, columns=[f"p_{c}" for c in classes]).assign(
        patient_id=pid, y_true=[classes[i] for i in yp]
    ).to_csv(out_dir / "predictions_patient.csv", index=False)
    save_figures(yp, pp, classes, out_dir, "paciente")
    return report
