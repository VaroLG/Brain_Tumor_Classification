"""Conversión de series DICOM a PNG (REMBRANDT y colecciones TCIA similares).

Diferencias con el notebook original (``Ordenar Rembrandt_II_III_IV``):

- El ``patient_id`` se lee de la cabecera DICOM (etiqueta PatientID), no del
  nombre de la carpeta. Así cada corte queda ligado a su paciente y el split
  posterior puede agruparlos.
- Se elige la serie por su descripción (``SeriesDescription``), p. ej. solo
  T1 post-contraste, en lugar de mezclar todas las secuencias del estudio.
- REMBRANDT no trae segmentaciones, así que no podemos elegir cortes con
  tumor automáticamente como en UPENN. Hay dos opciones:
    * ``selected_slices``: CSV con los cortes elegidos a mano (patient_id,
      instance_number). Permite reutilizar la selección manual del TFM, pero
      ahora documentada y versionable.
    * por defecto, la franja central del volumen (``central_fraction``), donde
      suelen estar los gliomas supratentoriales. Es una heurística: se indica
      claramente como limitación en el README.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from btc.data.preprocessing import crop_to_foreground, normalize_to_uint8, save_png
from btc.utils import get_logger

log = get_logger(__name__)


def _series_matches(description: str, include: list[str], exclude: list[str]) -> bool:
    d = description.upper()
    return all(k.upper() in d for k in include) and not any(k.upper() in d for k in exclude)


def build_dicom_manifest(
    dicom_root: str | Path,
    labels: pd.DataFrame,
    out_dir: str | Path,
    series_include: list[str] | None = None,
    series_exclude: list[str] | None = None,
    central_fraction: float = 0.4,
    max_slices: int = 10,
    selected_slices: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Lee todos los .dcm, agrupa por (paciente, serie) y guarda los cortes elegidos.

    Parameters
    ----------
    labels : ``patient_id, label`` (p. ej. grado II/III/IV desde la clínica).
    series_include / series_exclude : palabras clave en SeriesDescription,
        p. ej. include=["T1"], exclude=["PRE"] para T1 post-contraste.
    central_fraction : fracción central del volumen a considerar si no hay
        selección manual.
    selected_slices : DataFrame opcional ``patient_id, instance_number``.
    """
    import pydicom  # dependencia opcional: pip install "btc[medical]"

    series_include = series_include or []
    series_exclude = series_exclude or []
    label_of = dict(zip(labels["patient_id"].astype(str), labels["label"], strict=True))
    manual = None
    if selected_slices is not None:
        manual = set(
            zip(
                selected_slices["patient_id"].astype(str),
                selected_slices["instance_number"].astype(int),
                strict=True,
            )
        )

    # 1) Indexar ficheros por serie leyendo solo cabeceras (rápido)
    series: dict[tuple[str, str], list[tuple[int, Path]]] = defaultdict(list)
    for path in Path(dicom_root).rglob("*.dcm"):
        hdr = pydicom.dcmread(path, stop_before_pixels=True)
        pid = str(getattr(hdr, "PatientID", ""))
        desc = str(getattr(hdr, "SeriesDescription", ""))
        if pid not in label_of or not _series_matches(desc, series_include, series_exclude):
            continue
        series[(pid, str(hdr.SeriesInstanceUID))].append((int(hdr.InstanceNumber), path))

    # 2) Por cada serie, ordenar por InstanceNumber y quedarse con los cortes elegidos
    rows = []
    for (pid, uid), items in series.items():
        items.sort()
        n = len(items)
        if manual is not None:
            chosen = [(i, p) for i, p in items if (pid, i) in manual]
        else:
            lo, hi = int(n * (0.5 - central_fraction / 2)), int(n * (0.5 + central_fraction / 2))
            window = items[lo : max(hi, lo + 1)]
            pick = np.linspace(0, len(window) - 1, min(max_slices, len(window))).round().astype(int)
            chosen = [window[k] for k in sorted(set(pick))]
        for inst, path in chosen:
            arr = pydicom.dcmread(path).pixel_array
            img = crop_to_foreground(normalize_to_uint8(arr))
            png = Path(out_dir) / label_of[pid] / f"{pid}_{uid[-8:]}_{inst:03d}.png"
            save_png(img, png)
            rows.append(
                {
                    "image_path": str(png),
                    "patient_id": pid,
                    "label": label_of[pid],
                    "series_uid": uid,
                    "instance_number": inst,
                }
            )

    manifest = pd.DataFrame(rows)
    log.info(
        "Manifest DICOM: %d imágenes de %d pacientes",
        len(manifest),
        manifest["patient_id"].nunique() if rows else 0,
    )
    return manifest
