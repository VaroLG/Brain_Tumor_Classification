"""Etiquetas clínicas y moleculares a nivel de paciente (UPENN-GBM).

Problema de la versión original con la supervivencia
----------------------------------------------------
La supervivencia global (SG) es un dato *censurado*: si un paciente seguía vivo
al final del estudio (o se perdió el seguimiento) solo sabemos que vivió **al
menos** X días, no cuántos. El TFM partía a los pacientes por la media (18
meses) sin mirar la censura, de modo que un paciente vivo con 6 meses de
seguimiento acababa en "SG baja" aunque no hubiera fallecido. Eso mete ruido
directamente en las etiquetas.

Regla aplicada aquí (umbral T):
- Falleció antes de T                 -> ``short``  (SG baja), sabemos que es cierto
- Seguimiento (vivo o no) >= T        -> ``long``   (SG alta), sabemos que es cierto
- Censurado antes de T                -> se excluye: no sabemos a qué grupo pertenece

Además se usa la **mediana** en vez de la media como umbral por defecto: es
robusta a los pocos supervivientes muy largos y da grupos más equilibrados.

Siguiente paso natural (ver docs/ROADMAP_RADIOGENOMICA.md): dejar de binarizar
y usar un modelo de supervivencia propiamente dicho (Cox, C-index), que
aprovecha también los casos censurados.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def survival_label(days: float, event: int | bool, threshold_days: float) -> str | None:
    """Etiqueta binaria de supervivencia teniendo en cuenta la censura.

    Parameters
    ----------
    days : días desde la cirugía hasta el fallecimiento o el último contacto.
    event : 1/True si el paciente falleció, 0/False si está censurado.
    threshold_days : umbral que separa SG baja de SG alta.

    Returns
    -------
    ``"short"``, ``"long"`` o ``None`` (excluir del análisis).
    """
    if days is None or (isinstance(days, float) and np.isnan(days)):
        return None
    if days >= threshold_days:
        return "long"
    return "short" if bool(event) else None


def build_survival_labels(
    clinical: pd.DataFrame,
    id_col: str,
    days_col: str,
    event_col: str | None = None,
    threshold_days: float | None = None,
) -> pd.DataFrame:
    """Devuelve un DataFrame ``patient_id, label`` con la SG binarizada.

    Si ``event_col`` es None se asume que todos fallecieron (todos eventos).
    Es lo que hacía implícitamente el TFM; se permite por compatibilidad, pero
    se recomienda indicar la columna de censura si existe en el CSV clínico.
    """
    df = clinical[[id_col, days_col] + ([event_col] if event_col else [])].copy()
    df[days_col] = pd.to_numeric(df[days_col], errors="coerce")
    df = df.dropna(subset=[days_col])
    events = df[event_col].astype(int) if event_col else pd.Series(1, index=df.index)
    if threshold_days is None:
        threshold_days = float(df[days_col].median())
    labels = [
        survival_label(d, e, threshold_days) for d, e in zip(df[days_col], events, strict=True)
    ]
    out = pd.DataFrame({"patient_id": df[id_col].astype(str), "label": labels})
    out.attrs["threshold_days"] = threshold_days
    return out.dropna(subset=["label"]).reset_index(drop=True)


def build_categorical_labels(
    clinical: pd.DataFrame, id_col: str, value_col: str, mapping: dict[str, str]
) -> pd.DataFrame:
    """Etiquetas moleculares (IDH1, MGMT...) a partir de una columna del CSV clínico.

    ``mapping`` traduce los valores del CSV a nombres de clase y, al mismo
    tiempo, actúa como lista blanca: todo valor que no esté en el mapping
    (p. ej. "Not Available", "Indeterminate") se descarta en lugar de
    asignarse por defecto a una clase. El original agrupaba "no metilado" y
    "no clasificado" en la misma carpeta U, mezclando desconocidos con negativos.

    Ejemplo para MGMT::

        mapping={"Methylated": "methylated", "Unmethylated": "unmethylated"}
    """
    values = clinical[value_col].astype(str).str.strip()
    labels = values.map(mapping)
    out = pd.DataFrame({"patient_id": clinical[id_col].astype(str), "label": labels})
    return out.dropna(subset=["label"]).reset_index(drop=True)
