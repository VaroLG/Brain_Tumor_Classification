"""Preprocesado de intensidades y recorte de fondo.

Sustituye a tres copias idénticas de ``crop_black_background`` (una por grado
de glioma) y a la normalización min-max de los scripts de extracción.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def normalize_to_uint8(
    img: np.ndarray, mask: np.ndarray | None = None, p_low: float = 0.5, p_high: float = 99.5
) -> np.ndarray:
    """Lleva un corte de RM a [0, 255] de forma robusta a valores atípicos.

    ¿Por qué no min-max puro (lo que hacía el código original)?
    Las intensidades de RM no tienen unidades absolutas y suelen tener unos
    pocos vóxeles muy brillantes (artefactos, vasos con contraste). Con min-max
    un solo vóxel extremo comprime todo el tejido útil en unos pocos niveles de
    gris. Recortar a los percentiles 0.5–99.5 antes de escalar es la práctica
    habitual en neuroimagen.

    ``mask`` (opcional) limita el cálculo de percentiles al cerebro, para que el
    fondo negro (que es la mayoría de píxeles) no sesgue los percentiles.

    Además se protege la división por cero: el original producía NaN -> basura
    en cortes completamente negros (max == min).
    """
    img = img.astype(np.float32)
    ref = img[mask > 0] if mask is not None and mask.any() else img[img > 0]
    if ref.size == 0:  # corte vacío
        return np.zeros(img.shape, dtype=np.uint8)
    lo, hi = np.percentile(ref, [p_low, p_high])
    if hi <= lo:
        return np.zeros(img.shape, dtype=np.uint8)
    img = np.clip((img - lo) / (hi - lo), 0.0, 1.0)
    return (img * 255).round().astype(np.uint8)


def crop_to_foreground(img: np.ndarray, threshold: int = 20, margin: int = 2) -> np.ndarray:
    """Recorta el marco negro alrededor del cráneo/cerebro.

    Misma idea que el notebook original (umbral de intensidad 20), pero:
    - en NumPy puro, sin depender de TensorFlow para algo tan simple;
    - acepta imágenes 2D (gris) o 3D (H, W, C);
    - deja un pequeño margen para no cortar justo en el borde del tejido.

    Si la imagen está vacía (todo por debajo del umbral) se devuelve sin cambios.
    """
    gray = img if img.ndim == 2 else img.mean(axis=-1)
    ys, xs = np.nonzero(gray > threshold)
    if ys.size == 0:
        return img
    y0, y1 = max(ys.min() - margin, 0), min(ys.max() + margin + 1, img.shape[0])
    x0, x1 = max(xs.min() - margin, 0), min(xs.max() + margin + 1, img.shape[1])
    return img[y0:y1, x0:x1]


def save_png(img: np.ndarray, path: str | Path) -> None:
    """Guarda un array uint8 como PNG creando las carpetas necesarias."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(path)
