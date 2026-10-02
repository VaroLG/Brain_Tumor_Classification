"""Interfaz de línea de comandos: ``btc <subcomando> ...``.

Flujo típico (ver README)::

    btc manifest-upenn --nifti-root data/UPENN-GBM \\
        --clinical data/UPENN-GBM_clinical_info_v2.1.csv --task survival \\
        --out-dir data/png/gbm_survival --manifest data/gbm_survival.csv
    btc split --manifest data/gbm_survival.csv
    btc train --config configs/gbm_survival_upenn.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from btc.utils import get_logger

log = get_logger("btc.cli")


# ----------------------------------------------------------------------------
# Construcción de manifests
# ----------------------------------------------------------------------------
def _cmd_manifest_folders(a: argparse.Namespace) -> None:
    from btc.data.folders import build_folder_manifest

    df = build_folder_manifest(a.root, group_by_hash=not a.no_hash)
    _write(df, a.manifest)


def _require_cols(df: pd.DataFrame, cols: list[str | None]) -> None:
    missing = [c for c in cols if c and c not in df.columns]
    if missing:
        raise SystemExit(
            f"Columnas {missing} no encontradas en el CSV clínico.\n"
            f"Columnas disponibles: {list(df.columns)}\n"
            "Indica las correctas con --id-col / --days-col / --event-col / --value-col."
        )


def _cmd_manifest_upenn(a: argparse.Namespace) -> None:
    from btc.data.labels import build_categorical_labels, build_survival_labels
    from btc.data.nifti import SliceSelection, build_nifti_manifest

    clinical = pd.read_csv(a.clinical)
    if a.task == "survival":
        _require_cols(clinical, [a.id_col, a.days_col, a.event_col])
        labels = build_survival_labels(
            clinical, a.id_col, a.days_col, a.event_col, a.threshold_days
        )
        log.info("Umbral de supervivencia: %.0f días", labels.attrs["threshold_days"])
    else:
        value_col = a.value_col or {"idh": "IDH1", "mgmt": "MGMT"}[a.task]
        _require_cols(clinical, [a.id_col, value_col])
        mapping = (
            json.loads(a.mapping)
            if a.mapping
            else {
                "idh": {"Mutated": "mutant", "Wildtype": "wildtype"},
                "mgmt": {"Methylated": "methylated", "Unmethylated": "unmethylated"},
            }[a.task]
        )
        labels = build_categorical_labels(clinical, a.id_col, value_col, mapping)
    log.info("Pacientes con etiqueta: %d\n%s", len(labels), labels["label"].value_counts())

    sel = SliceSelection(
        axis=a.axis, min_tumor_fraction=a.min_tumor_fraction, max_slices=a.max_slices
    )
    df = build_nifti_manifest(
        a.nifti_root, labels, a.out_dir, modality=a.modality, visits=tuple(a.visits), selection=sel
    )
    _write(df, a.manifest)


def _cmd_manifest_dicom(a: argparse.Namespace) -> None:
    from btc.data.dicom import build_dicom_manifest

    labels = pd.read_csv(a.labels)
    _require_cols(labels, ["patient_id", "label"])
    selected = pd.read_csv(a.selected_slices) if a.selected_slices else None
    df = build_dicom_manifest(
        a.dicom_root,
        labels,
        a.out_dir,
        a.include,
        a.exclude,
        a.central_fraction,
        a.max_slices,
        selected,
    )
    _write(df, a.manifest)


# ----------------------------------------------------------------------------
# Split, entrenamiento y evaluación
# ----------------------------------------------------------------------------
def _cmd_split(a: argparse.Namespace) -> None:
    from btc.splits import patient_level_split, split_summary

    df = pd.read_csv(a.manifest)
    out = patient_level_split(df, a.val, a.test, seed=a.seed)
    log.info("Split por paciente:\n%s", split_summary(out))
    _write(out, a.out or a.manifest)


def _cmd_train(a: argparse.Namespace) -> None:
    from btc.config import ExperimentConfig
    from btc.train import train

    train(ExperimentConfig.from_yaml(a.config), a.run_dir)


def _cmd_evaluate(a: argparse.Namespace) -> None:
    """Re-evalúa un run guardado (útil si cambian las métricas o las figuras)."""
    import keras

    from btc.config import ExperimentConfig
    from btc.datasets import make_dataset
    from btc.evaluate import evaluate_predictions

    run = Path(a.run_dir)
    cfg = ExperimentConfig.from_yaml(run / "config.yaml")
    test = pd.read_csv(run / "manifest.csv").query("split == 'test'").reset_index(drop=True)
    model = keras.models.load_model(run / "model.keras")
    ds = make_dataset(
        test, cfg.data.classes, cfg.model.backbone, cfg.data.image_size, cfg.data.batch_size
    )
    probs = model.predict(ds, verbose=0)
    y = test["label"].map({c: i for i, c in enumerate(cfg.data.classes)}).to_numpy()
    rep = evaluate_predictions(test["patient_id"].to_numpy(), y, probs, cfg.data.classes, run)
    print(json.dumps(rep["patient_level"], indent=2))


def _write(df: pd.DataFrame, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    log.info("Manifest escrito en %s (%d filas)", path, len(df))


# ----------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="btc", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("manifest-folders", help="Dataset en carpetas por clase (Kaggle)")
    s.add_argument("--root", required=True)
    s.add_argument("--manifest", required=True)
    s.add_argument("--no-hash", action="store_true", help="No agrupar casi-duplicados por dHash")
    s.set_defaults(func=_cmd_manifest_folders)

    s = sub.add_parser("manifest-upenn", help="UPENN-GBM (NIfTI + CSV clínico)")
    s.add_argument("--nifti-root", required=True)
    s.add_argument("--clinical", required=True)
    s.add_argument("--task", choices=["survival", "idh", "mgmt"], required=True)
    s.add_argument("--out-dir", required=True, help="Carpeta donde guardar los PNG")
    s.add_argument("--manifest", required=True)
    # Nombres de columna: verifica con `head -1` de tu CSV clínico; cambian entre versiones
    s.add_argument("--id-col", default="ID")
    s.add_argument("--days-col", default="Survival_from_surgery_days_UPDATED")
    s.add_argument(
        "--event-col",
        default=None,
        help="Columna 1=fallecido/0=censurado. Sin ella se asume que todos fallecieron",
    )
    s.add_argument("--threshold-days", type=float, default=None, help="Por defecto, la mediana")
    s.add_argument("--value-col", default=None, help="Columna molecular (IDH1/MGMT)")
    s.add_argument(
        "--mapping", default=None, help='JSON valor->clase, p. ej. \'{"Mutated":"mutant"}\''
    )
    s.add_argument("--modality", default="T1GD", choices=["T1", "T1GD", "T2", "FLAIR"])
    s.add_argument("--axis", default="axial", choices=["axial", "coronal", "sagittal"])
    s.add_argument("--visits", nargs="+", default=["11"])
    s.add_argument("--min-tumor-fraction", type=float, default=0.25)
    s.add_argument("--max-slices", type=int, default=10)
    s.set_defaults(func=_cmd_manifest_upenn)

    s = sub.add_parser("manifest-dicom", help="Series DICOM (REMBRANDT)")
    s.add_argument("--dicom-root", required=True)
    s.add_argument("--labels", required=True, help="CSV con patient_id,label")
    s.add_argument("--out-dir", required=True)
    s.add_argument("--manifest", required=True)
    s.add_argument("--include", nargs="*", default=["T1"])
    s.add_argument("--exclude", nargs="*", default=["PRE"])
    s.add_argument("--central-fraction", type=float, default=0.4)
    s.add_argument("--max-slices", type=int, default=10)
    s.add_argument(
        "--selected-slices",
        default=None,
        help="CSV patient_id,instance_number con cortes elegidos a mano",
    )
    s.set_defaults(func=_cmd_manifest_dicom)

    s = sub.add_parser("split", help="Split train/val/test por paciente")
    s.add_argument("--manifest", required=True)
    s.add_argument("--out", default=None, help="Por defecto sobrescribe el manifest")
    s.add_argument("--val", type=float, default=0.15)
    s.add_argument("--test", type=float, default=0.20)
    s.add_argument("--seed", type=int, default=42)
    s.set_defaults(func=_cmd_split)

    s = sub.add_parser("train", help="Entrenar y evaluar un experimento")
    s.add_argument("--config", required=True)
    s.add_argument("--run-dir", default=None)
    s.set_defaults(func=_cmd_train)

    s = sub.add_parser("evaluate", help="Re-evaluar un run guardado")
    s.add_argument("--run-dir", required=True)
    s.set_defaults(func=_cmd_evaluate)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
