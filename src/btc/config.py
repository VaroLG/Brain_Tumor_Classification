"""Configuración de experimentos en YAML.

Antes, cada hiperparámetro estaba escrito a mano en una celda distinta de cada
notebook (y la tasa de aprendizaje anotada en un markdown no siempre coincidía
con la del código). Ahora un experimento queda descrito por completo en un
fichero ``configs/*.yaml`` que se copia dentro de la carpeta del run: cualquier
resultado se puede rastrear hasta la configuración exacta que lo produjo.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    manifest: str  # CSV con image_path, patient_id, label y split
    classes: list[str]  # orden fijo de clases -> índice 0..n-1
    image_size: int = 240
    batch_size: int = 32


@dataclass
class ModelConfig:
    backbone: str = "efficientnetb1"  # ver btc.models.BACKBONES
    dense_units: list[int] = field(default_factory=lambda: [256])
    dropout: float = 0.3
    weights: str | None = "imagenet"  # None -> pesos aleatorios (útil en tests)


@dataclass
class TrainConfig:
    seed: int = 42
    # Fase 1: backbone congelado, solo se entrena la cabeza nueva
    head_epochs: int = 15
    head_lr: float = 1e-3
    # Fase 2: fine-tuning de las últimas capas del backbone con LR pequeña
    finetune_epochs: int = 15
    finetune_lr: float = 1e-5
    finetune_layers: int = 30  # nº de capas finales a descongelar (0 = sin fase 2)
    early_stopping_patience: int = 5
    class_weights: bool = True  # compensa clases desbalanceadas
    augment: bool = True


@dataclass
class ExperimentConfig:
    name: str
    data: DataConfig
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    output_dir: str = "runs"

    @classmethod
    def from_yaml(cls, path: str | Path) -> ExperimentConfig:
        with open(path, encoding="utf-8") as fh:
            raw: dict[str, Any] = yaml.safe_load(fh)
        return cls(
            name=raw["name"],
            data=DataConfig(**raw["data"]),
            model=ModelConfig(**raw.get("model", {})),
            train=TrainConfig(**raw.get("train", {})),
            output_dir=raw.get("output_dir", "runs"),
        )

    def to_yaml(self, path: str | Path) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            yaml.safe_dump(asdict(self), fh, sort_keys=False, allow_unicode=True)
