"""Manifest a partir de un dataset organizado en carpetas por clase (Kaggle).

El dataset de Kaggle "Brain Tumor Classification (MRI)" tiene la forma::

    Training/glioma_tumor/*.jpg   Testing/glioma_tumor/*.jpg
    Training/meningioma_tumor/... ...

Problema: **no trae identificador de paciente**. Varias imágenes son cortes
del mismo estudio o incluso duplicados exactos/casi exactos (reescalados,
recomprimidos), y se ha documentado que algunas se repiten entre Training y
Testing. Sin ID de paciente no podemos garantizar un split sin fuga.

Mitigación aplicada: agrupamos imágenes casi idénticas mediante un *hash
perceptual* (dHash). Imágenes con el mismo hash forman un "pseudo-paciente"
y van siempre al mismo split. No sustituye a un ID real (dos cortes distintos
del mismo paciente tienen hashes distintos), así que las métricas sobre este
dataset deben leerse como optimistas. Queda indicado en el README.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from btc.utils import get_logger

log = get_logger(__name__)

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def dhash(path: str | Path, size: int = 8) -> str:
    """Hash perceptual por diferencias (dHash) de 64 bits en hexadecimal.

    Cómo funciona: se reduce la imagen a (size+1) x size en gris y se compara
    cada píxel con su vecino de la derecha (1 si es más brillante). El patrón de
    bits resume la estructura gruesa de la imagen: sobrevive a reescalados y
    recompresión JPEG, así que detecta duplicados que un hash MD5 no vería.
    """
    with Image.open(path) as im:
        small = np.asarray(im.convert("L").resize((size + 1, size), Image.Resampling.LANCZOS),
                           dtype=np.int16)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    return f"{int(''.join('1' if b else '0' for b in bits), 2):0{size * size // 4}x}"


def build_folder_manifest(root: str | Path, group_by_hash: bool = True) -> pd.DataFrame:
    """Recorre ``root/**/<clase>/<imagen>`` y devuelve el manifest.

    La clase es el nombre de la carpeta que contiene la imagen. Se ignoran las
    particiones Training/Testing originales: se rehace el split por grupos.
    """
    rows = []
    for path in sorted(Path(root).rglob("*")):
        if path.suffix.lower() in IMAGE_EXTS:
            rows.append({"image_path": str(path), "label": path.parent.name})
    df = pd.DataFrame(rows)
    if df.empty:
        raise FileNotFoundError(f"No se encontraron imágenes en {root}")

    if group_by_hash:
        df["patient_id"] = "h_" + df["image_path"].map(dhash)
        n_dup = len(df) - df["patient_id"].nunique()
        log.info("dHash: %d imágenes en %d grupos (%d casi-duplicados agrupados)",
                 len(df), df["patient_id"].nunique(), n_dup)
        # Un mismo hash con dos etiquetas distintas indica un duplicado mal etiquetado
        conflicts = df.groupby("patient_id")["label"].nunique()
        if (conflicts > 1).any():
            log.warning("%d grupos con etiquetas contradictorias", int((conflicts > 1).sum()))
    else:
        df["patient_id"] = df["image_path"]  # cada imagen es su propio grupo
    return df
