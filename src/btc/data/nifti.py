"""Extracción de cortes 2D desde volúmenes NIfTI (UPENN-GBM y similares, formato BraTS).

Qué cambia respecto al notebook original
----------------------------------------
1. **Selección de cortes con la máscara de segmentación, no a mano.**
   El TFM extraía TODOS los cortes (40.424 imágenes T1 axiales), los filtraba
   por tamaño de fichero (< 10 KB = descartar) y después se escogían a ojo las
   imágenes "donde se apreciaba la masa tumoral". Eso es lento, no reproducible
   y subjetivo. UPENN-GBM ya trae segmentaciones del tumor (manuales y
   automáticas): basta con quedarse con los cortes cuya área tumoral supere un
   umbral. Mismo criterio para todos los pacientes, en segundos.

2. **Un identificador de paciente por corte.** El nombre de sujeto en UPENN es
   ``UPENN-GBM-00001_11``: ``_11`` es la RM preoperatoria y ``_21`` una de
   seguimiento. Ambas son el MISMO paciente, así que ``patient_id`` se queda
   con ``UPENN-GBM-00001`` para que nunca caigan en splits distintos.

3. **Por defecto solo la RM preoperatoria (``_11``)**, que es la que tiene
   sentido para predecir pronóstico/genética antes del tratamiento.

4. **T1-Gd en lugar de T1 por defecto.** El TFM usó T1 sin contraste; para
   GBM la secuencia con gadolinio (realce del anillo tumoral) y FLAIR (edema)
   son las más informativas. Se puede cambiar por configuración.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from btc.data.preprocessing import crop_to_foreground, normalize_to_uint8, save_png
from btc.utils import get_logger

log = get_logger(__name__)

AXES = {"sagittal": 0, "coronal": 1, "axial": 2}
_SUBJECT_RE = re.compile(r"^(?P<patient>.+?)_(?P<visit>\d{2})$")


@dataclass
class SliceSelection:
    """Criterio para decidir qué cortes de un volumen se guardan."""

    axis: str = "axial"
    min_tumor_fraction: float = 0.25  # área tumoral mínima relativa al corte con más tumor
    max_slices: int = 10  # tope por paciente (evita que pacientes grandes dominen)


def split_subject_id(subject: str) -> tuple[str, str]:
    """``UPENN-GBM-00001_11`` -> (``UPENN-GBM-00001``, ``11``)."""
    m = _SUBJECT_RE.match(subject)
    if not m:
        return subject, ""
    return m.group("patient"), m.group("visit")


def select_slice_indices(mask: np.ndarray, sel: SliceSelection) -> list[int]:
    """Índices de los cortes con suficiente tumor, muestreados de forma uniforme.

    Paso a paso:
    1. Área tumoral (nº de vóxeles != 0) de cada corte a lo largo del eje elegido.
    2. Se descartan cortes con menos de ``min_tumor_fraction`` x área máxima:
       los extremos del tumor son casi todo cerebro sano y aportan poca señal.
    3. Si quedan más de ``max_slices``, se toman ``max_slices`` equiespaciados
       (no los consecutivos, que serían casi duplicados).
    """
    ax = AXES[sel.axis]
    other = tuple(i for i in range(3) if i != ax)
    area = (mask > 0).sum(axis=other)
    if area.max() == 0:
        return []
    candidates = np.flatnonzero(area >= sel.min_tumor_fraction * area.max())
    if len(candidates) > sel.max_slices:
        pick = np.linspace(0, len(candidates) - 1, sel.max_slices).round().astype(int)
        candidates = candidates[pick]
    return candidates.tolist()


def take_slice(volume: np.ndarray, axis: str, idx: int) -> np.ndarray:
    """Extrae un corte y lo rota 90º para verlo en orientación radiológica habitual."""
    sl = np.take(volume, idx, axis=AXES[axis])
    return np.rot90(sl)


def iter_slices(
    image: np.ndarray, mask: np.ndarray, sel: SliceSelection
) -> Iterator[tuple[int, np.ndarray]]:
    """Genera (índice, corte uint8 normalizado y recortado) para los cortes elegidos."""
    for idx in select_slice_indices(mask, sel):
        raw = take_slice(image, sel.axis, idx)
        brain = raw > 0  # UPENN ya viene sin cráneo: fuera del cerebro es 0
        yield idx, crop_to_foreground(normalize_to_uint8(raw, mask=brain))


def _find_one(root: Path, pattern: str) -> Path | None:
    hits = sorted(root.glob(pattern))
    return hits[0] if hits else None


def build_nifti_manifest(
    nifti_root: str | Path,
    labels: pd.DataFrame,
    out_dir: str | Path,
    modality: str = "T1GD",
    image_glob: str = "**/{subject}_{modality}.nii.gz",
    mask_globs: tuple[str, ...] = (
        "**/{subject}_segm.nii.gz",  # segmentación revisada manualmente
        "**/{subject}_automated_approx_segm.nii.gz",  # automática (resto de sujetos)
    ),
    visits: tuple[str, ...] = ("11",),
    selection: SliceSelection | None = None,
) -> pd.DataFrame:
    """Recorre los sujetos, guarda los cortes seleccionados como PNG y devuelve el manifest.

    Parameters
    ----------
    nifti_root : carpeta raíz de la descarga NIfTI de TCIA.
    labels : DataFrame ``patient_id, label`` (ver ``btc.data.labels``). Solo se
        procesan pacientes con etiqueta.
    out_dir : donde se escriben los PNG (``out_dir/<label>/<subject>_<idx>.png``).
    modality : sufijo de la secuencia en el nombre de fichero (T1, T1GD, T2, FLAIR).
    image_glob, mask_globs : patrones relativos a ``nifti_root``. Los valores por
        defecto siguen la estructura publicada de UPENN-GBM; si tu descarga es
        distinta, ajústalos sin tocar el código.
    visits : visitas a incluir (``"11"`` = preoperatoria).
    """
    import nibabel as nib  # dependencia opcional: pip install "btc[medical]"

    nifti_root, out_dir = Path(nifti_root), Path(out_dir)
    selection = selection or SliceSelection()
    label_of = dict(zip(labels["patient_id"].astype(str), labels["label"], strict=True))

    subjects = sorted(
        {
            p.name.split(f"_{modality}")[0]
            for p in nifti_root.glob(image_glob.format(subject="*", modality=modality))
        }
    )
    rows, skipped = [], 0
    for subject in subjects:
        patient, visit = split_subject_id(subject)
        if visit not in visits or patient not in label_of:
            continue
        img_path = _find_one(nifti_root, image_glob.format(subject=subject, modality=modality))
        mask_path = next(
            (p for g in mask_globs if (p := _find_one(nifti_root, g.format(subject=subject)))),
            None,
        )
        if img_path is None or mask_path is None:
            skipped += 1
            log.warning("Sin imagen o máscara para %s: se omite", subject)
            continue

        image = np.asarray(nib.load(img_path).dataobj, dtype=np.float32)
        mask = np.asarray(nib.load(mask_path).dataobj)
        if image.shape != mask.shape:
            skipped += 1
            log.warning(
                "Forma distinta imagen %s vs máscara %s en %s", image.shape, mask.shape, subject
            )
            continue

        label = label_of[patient]
        for idx, sl in iter_slices(image, mask, selection):
            png = out_dir / label / f"{subject}_{modality}_{selection.axis}_{idx:03d}.png"
            save_png(sl, png)
            rows.append(
                {
                    "image_path": str(png),
                    "patient_id": patient,
                    "label": label,
                    "subject": subject,
                    "slice_idx": idx,
                }
            )

    manifest = pd.DataFrame(rows)
    log.info(
        "Manifest NIfTI: %d imágenes de %d pacientes (%d sujetos omitidos)",
        len(manifest),
        manifest["patient_id"].nunique() if rows else 0,
        skipped,
    )
    return manifest
