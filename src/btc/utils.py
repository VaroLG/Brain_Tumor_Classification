"""Utilidades transversales: semillas y logging."""

from __future__ import annotations

import logging
import os
import random

import numpy as np


def set_seed(seed: int) -> None:
    """Fija todas las fuentes de aleatoriedad para que un experimento sea repetible.

    ¿Por qué hace falta? En los notebooks originales ``random.shuffle`` decidía
    qué imagen iba a train/val/test y TensorFlow inicializaba los pesos al azar
    sin semilla: dos ejecuciones idénticas daban métricas distintas y era
    imposible saber si un cambio de hiperparámetros mejoraba algo o era ruido.

    Nota: en GPU algunas operaciones de cuDNN no son deterministas incluso con
    semilla. Si se necesita reproducibilidad bit a bit, activar además
    ``tf.config.experimental.enable_op_determinism()`` (más lento).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 - semilla global que usan librerías de terceros
    try:  # TensorFlow es opcional para las partes de preprocesado/split
        import tensorflow as tf

        tf.random.set_seed(seed)
    except ImportError:  # pragma: no cover
        pass


def get_logger(name: str = "btc") -> logging.Logger:
    """Logger con formato homogéneo (sustituye a los ``print`` sueltos)."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s", "%H:%M:%S")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
