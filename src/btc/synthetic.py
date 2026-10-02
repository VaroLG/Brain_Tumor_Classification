"""Datos sintéticos para tests y para probar el pipeline sin descargar nada.

Genera "cortes" 240x240 con forma de cerebro (elipse con textura) y una lesión
brillante cuyo tamaño depende de la clase, de modo que un modelo SÍ puede
aprender algo. Cada paciente sintético aporta varios cortes parecidos entre sí,
igual que en los datos reales: así los tests ejercitan la lógica de split por
paciente y de agregación por paciente.

IMPORTANTE: esto no es una simulación de RM realista; solo sirve para
comprobar que el código funciona de punta a punta.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from btc.data.preprocessing import save_png


def fake_slice(rng: np.random.Generator, lesion_radius: float, size: int = 240) -> np.ndarray:
    yy, xx = np.mgrid[:size, :size]
    cy, cx = size / 2 + rng.normal(0, 3), size / 2 + rng.normal(0, 3)
    brain = ((yy - cy) / (size * 0.38)) ** 2 + ((xx - cx) / (size * 0.30)) ** 2 <= 1
    img = np.where(brain, 90 + rng.normal(0, 12, (size, size)), 0)
    ly, lx = cy + rng.normal(0, 15), cx + rng.normal(0, 15)
    lesion = (yy - ly) ** 2 + (xx - lx) ** 2 <= lesion_radius**2
    img = np.where(lesion & brain, 220 + rng.normal(0, 10, (size, size)), img)
    return np.clip(img, 0, 255).astype(np.uint8)


def make_synthetic_manifest(
    out_dir: str | Path,
    classes: tuple[str, ...] = ("small", "large"),
    patients_per_class: int = 12,
    slices_per_patient: int = 4,
    size: int = 240,
    seed: int = 0,
) -> pd.DataFrame:
    """Escribe PNGs en ``out_dir/<clase>/`` y devuelve el manifest (sin split)."""
    rng = np.random.default_rng(seed)
    rows = []
    for ci, cls in enumerate(classes):
        base_radius = 8 + 14 * ci  # la clase determina el tamaño de la lesión
        for p in range(patients_per_class):
            pid = f"{cls}_P{p:03d}"
            radius = base_radius + rng.normal(0, 2)
            for s in range(slices_per_patient):
                path = Path(out_dir) / cls / f"{pid}_{s:02d}.png"
                save_png(fake_slice(rng, max(radius + rng.normal(0, 1), 2), size), path)
                rows.append({"image_path": str(path), "patient_id": pid, "label": cls})
    return pd.DataFrame(rows)
