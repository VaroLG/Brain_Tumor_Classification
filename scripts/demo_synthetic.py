"""Demo de punta a punta SIN descargar datos reales.

Genera un dataset sintético, hace el split por paciente, entrena un modelo
pequeño y lo evalúa. Sirve para comprobar que la instalación funciona antes de
dedicar horas a descargar UPENN-GBM o REMBRANDT.

    python scripts/demo_synthetic.py            # pesos aleatorios, ~1 min en CPU
    python scripts/demo_synthetic.py --imagenet # con pesos de ImageNet (descarga ~30 MB)
"""

from __future__ import annotations

import argparse
from pathlib import Path

from btc.config import DataConfig, ExperimentConfig, ModelConfig, TrainConfig
from btc.splits import patient_level_split, split_summary
from btc.synthetic import make_synthetic_manifest
from btc.train import train


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/demo_synthetic")
    ap.add_argument("--imagenet", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)

    # 1) Datos sintéticos: 2 clases (lesión pequeña / grande), 20 pacientes por clase
    df = make_synthetic_manifest(out / "data", patients_per_class=20, slices_per_patient=4, size=64)
    # 2) Split por paciente
    df = patient_level_split(df, val_frac=0.2, test_frac=0.2, seed=0)
    print(split_summary(df))
    df.to_csv(out / "manifest.csv", index=False)

    # 3) Entrenar + evaluar
    # VGG19 porque no tiene BatchNormalization: entrenada desde cero con tan
    # pocos pasos por época, la BatchNorm de EfficientNet no llega a estimar sus
    # estadísticas y la validación se queda en azar. Con pesos de ImageNet
    # (--imagenet) ese problema desaparece y cualquier backbone sirve.
    cfg = ExperimentConfig(
        name="demo_synthetic",
        data=DataConfig(
            str(out / "manifest.csv"), ["small", "large"], image_size=64, batch_size=16
        ),
        model=ModelConfig("vgg19", [32], 0.2, "imagenet" if args.imagenet else None),
        train=TrainConfig(head_epochs=25, head_lr=1e-4, early_stopping_patience=8),
    )
    run = train(cfg, run_dir=out / "run")
    print(f"\nListo. Métricas y figuras en {run}")


if __name__ == "__main__":
    main()
