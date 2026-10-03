"""Características de imagen a partir de segmentaciones con etiquetas BraTS.

Qué significan las etiquetas
----------------------------
Las segmentaciones BraTS dividen el tumor en subregiones:

    1  NCR  núcleo necrótico / tumor no realzante (zona oscura en T1-Gd)
    2  ED   edema peritumoral (brillante en FLAIR)
    4  ET   tumor que realza con gadolinio (anillo brillante en T1-Gd)
           (en BraTS 2023+ la etiqueta del realce pasó a ser 3)

Por qué la fracción necrótica del NÚCLEO y no del tumor entero
--------------------------------------------------------------
El edema depende de muchos factores ajenos a la hipoxia (localización,
corticoides, tamaño...). Si lo metemos en el denominador, la variable mezcla
"cuánto tumor está necrótico" con "cuánto edema hay alrededor". Restringirnos
al núcleo (NCR + ET) aísla la pregunta: de la masa tumoral propiamente dicha,
¿qué proporción ha muerto?
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from btc.radiogenomics.cohorts import normalize_patient_id
from btc.utils import get_logger

log = get_logger("btc.rg.imaging")

LABEL_NCR = 1
LABEL_ED = 2
LABELS_ET = (3, 4)  # 4 en BraTS clásico, 3 en BraTS >= 2023


def region_volumes(mask: np.ndarray, voxel_volume_mm3: float = 1.0) -> dict[str, float]:
    """Volúmenes (mm³) de cada subregión y fracciones derivadas.

    Parameters
    ----------
    mask : array 3D de enteros con etiquetas BraTS.
    voxel_volume_mm3 : volumen de un vóxel. BraTS está remuestreado a 1 mm³
        isotrópico, así que por defecto 1; con otras máscaras se lee del NIfTI.

    Returns
    -------
    dict con ``v_ncr``, ``v_ed``, ``v_et``, ``v_total``, ``necrosis_fraction_core``
    (exposición primaria), ``necrosis_fraction_total`` y ``log_v_ncr``.
    """
    mask = np.asarray(mask)
    v_ncr = float((mask == LABEL_NCR).sum()) * voxel_volume_mm3
    v_ed = float((mask == LABEL_ED).sum()) * voxel_volume_mm3
    v_et = float(np.isin(mask, LABELS_ET).sum()) * voxel_volume_mm3
    v_core = v_ncr + v_et
    v_total = v_core + v_ed
    unknown = set(np.unique(mask)) - {0, LABEL_NCR, LABEL_ED, *LABELS_ET}
    if unknown:
        raise ValueError(f"Etiquetas no BraTS en la máscara: {sorted(unknown)}")
    return {
        "v_ncr": v_ncr,
        "v_ed": v_ed,
        "v_et": v_et,
        "v_total": v_total,
        # NaN (no 0) si no hay núcleo: "sin tumor" no es lo mismo que "sin necrosis"
        "necrosis_fraction_core": v_ncr / v_core if v_core > 0 else np.nan,
        "necrosis_fraction_total": v_ncr / v_total if v_total > 0 else np.nan,
        "log_v_ncr": float(np.log1p(v_ncr)),
        "log_v_total": float(np.log1p(v_total)),
    }


def _raw_id(path: Path, id_from_name: str) -> str:
    """Extrae el identificador "en bruto" de una máscara según la convención.

    - ``prefix``: inicio del nombre hasta el primer ``_``
      (``TCGA-02-0006_..._GlistrBoost...`` → ``TCGA-02-0006``).
    - ``parent``: nombre de la carpeta que contiene la máscara.
    - ``stem``: nombre sin extensión(es) ni el sufijo ``_seg``
      (``BraTS20_Training_001_seg.nii`` → ``BraTS20_Training_001``). Es lo que
      hace falta con BraTS 2020/2021, cuyos IDs llevan ``_`` dentro.
    """
    if id_from_name == "parent":
        return path.parent.name
    if id_from_name == "stem":
        name = path.name
        for ext in (".nii.gz", ".nii"):
            if name.endswith(ext):
                name = name[: -len(ext)]
        return name.removesuffix("_seg")
    return path.name.split("_")[0]


def load_id_map(path: str | Path, source_col: str, target_col: str) -> dict[str, str]:
    """Lee una tabla de equivalencias de IDs (p. ej. ``name_mapping.csv`` de BraTS 2020).

    Devuelve ``{id_brats: id_tcga}`` solo para las filas con destino conocido:
    en BraTS 2020 la mayoría de casos NO vienen de TCIA y tienen el destino vacío.
    """
    df = pd.read_csv(path)
    missing = [c for c in (source_col, target_col) if c not in df.columns]
    if missing:
        raise KeyError(f"Columnas {missing} no están en {path}. Disponibles: {list(df.columns)}")
    df = df[[source_col, target_col]].dropna()
    df = df[df[target_col].astype(str).str.strip() != ""]
    return dict(zip(df[source_col].astype(str), df[target_col].astype(str), strict=True))


def imaging_table(
    seg_root: str | Path,
    pattern: str = "**/*_GlistrBoost_ManuallyCorrected.nii.gz",
    cohort: str = "tcga",
    id_from_name: str = "prefix",
    id_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Recorre las máscaras y devuelve una fila de características por paciente.

    Parameters
    ----------
    seg_root : carpeta raíz de las segmentaciones.
    pattern : glob de los ficheros de máscara. El valor por defecto corresponde a
        las máscaras revisadas manualmente de BraTS-TCGA-GBM; para BraTS 2020 usar
        ``"**/*_seg.nii"`` y para BraTS 2021 ``"**/*_seg.nii.gz"``. Verifica el
        nombre real en tu descarga.
    cohort : ``"tcga"`` o ``"cptac"``; decide cómo se normaliza el ID.
    id_from_name : ``"prefix"``, ``"parent"`` o ``"stem"`` (ver ``_raw_id``).
    id_map : equivalencias ``{id_en_el_fichero: id_del_paciente}``. Si se da,
        las máscaras cuyo ID no aparece se descartan (con aviso del número): es
        el caso de los sujetos de BraTS que no proceden de TCGA.

    Si un paciente tiene varias máscaras se lanza un error en lugar de elegir
    una en silencio: hay que decidir explícitamente cuál es la preoperatoria.
    """
    import nibabel as nib

    rows = []
    n_unmapped = 0
    for path in sorted(Path(seg_root).glob(pattern)):
        raw_id = _raw_id(path, id_from_name)
        if id_map is not None:
            if raw_id not in id_map:
                n_unmapped += 1
                continue
            raw_id = id_map[raw_id]
        img = nib.load(path)
        voxel = float(np.prod(img.header.get_zooms()[:3]))
        feats = region_volumes(np.asarray(img.dataobj), voxel)
        rows.append(
            {"patient_id": normalize_patient_id(raw_id, cohort), "mask_path": str(path), **feats}
        )
    if not rows:
        raise FileNotFoundError(
            f"Ninguna máscara coincide con '{pattern}' en {seg_root}"
            + (f" ({n_unmapped} descartadas por no estar en la tabla de IDs)" if n_unmapped else "")
        )
    df = pd.DataFrame(rows)
    if n_unmapped:
        log.info("%d máscaras descartadas: su ID no está en la tabla de equivalencias", n_unmapped)
    dup = df["patient_id"][df["patient_id"].duplicated()].unique()
    if len(dup):
        raise ValueError(f"Pacientes con varias máscaras (elige una): {list(dup)[:5]}")
    return df.set_index("patient_id")
