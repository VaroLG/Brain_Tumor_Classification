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


def imaging_table(
    seg_root: str | Path,
    pattern: str = "**/*_GlistrBoost_ManuallyCorrected.nii.gz",
    cohort: str = "tcga",
    id_from_name: str = "prefix",
) -> pd.DataFrame:
    """Recorre las máscaras y devuelve una fila de características por paciente.

    Parameters
    ----------
    seg_root : carpeta raíz de las segmentaciones.
    pattern : glob de los ficheros de máscara. El valor por defecto corresponde a
        las máscaras revisadas manualmente de BraTS-TCGA-GBM; para BraTS 2021
        usar ``"**/*_seg.nii.gz"``. Verifica el nombre real en tu descarga.
    cohort : ``"tcga"`` o ``"cptac"``; decide cómo se normaliza el ID.
    id_from_name : ``"prefix"`` toma el ID del inicio del nombre de fichero;
        ``"parent"`` del nombre de la carpeta que lo contiene.

    Si un paciente tiene varias máscaras se lanza un error en lugar de elegir
    una en silencio: hay que decidir explícitamente cuál es la preoperatoria.
    """
    import nibabel as nib

    rows = []
    for path in sorted(Path(seg_root).glob(pattern)):
        raw_id = path.parent.name if id_from_name == "parent" else path.name.split("_")[0]
        img = nib.load(path)
        voxel = float(np.prod(img.header.get_zooms()[:3]))
        feats = region_volumes(np.asarray(img.dataobj), voxel)
        rows.append(
            {"patient_id": normalize_patient_id(raw_id, cohort), "mask_path": str(path), **feats}
        )
    if not rows:
        raise FileNotFoundError(f"Ninguna máscara coincide con '{pattern}' en {seg_root}")
    df = pd.DataFrame(rows)
    dup = df["patient_id"][df["patient_id"].duplicated()].unique()
    if len(dup):
        raise ValueError(f"Pacientes con varias máscaras (elige una): {list(dup)[:5]}")
    return df.set_index("patient_id")
