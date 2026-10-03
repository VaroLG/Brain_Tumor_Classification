"""División train / val / test **por paciente**.

Este módulo corrige el problema metodológico más grave de los notebooks
originales.

El problema: fuga de datos (data leakage)
-----------------------------------------
De un mismo volumen 3D se extraen muchos cortes 2D. Dos cortes axiales
contiguos de un paciente son prácticamente la misma imagen. El código original
hacía ``random.shuffle`` sobre la lista de *imágenes* y la partía 60/15/25, así
que cortes vecinos del mismo paciente acababan a la vez en train y en test.

El modelo puede entonces "acertar" en test reconociendo al paciente (la forma
de su cráneo, su tumor concreto) en lugar de aprender el patrón de la clase. La
métrica de test deja de medir generalización a pacientes nuevos, que es lo
único que importa clínicamente.

La solución: agrupar por paciente
---------------------------------
Todas las imágenes de un paciente van al mismo subconjunto. Usamos
``StratifiedGroupKFold`` de scikit-learn, que además intenta mantener la
proporción de clases en cada partición (estratificación).
"""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

SPLITS = ("train", "val", "test")


def _group_fold_split(
    df: pd.DataFrame, holdout_frac: float, group_col: str, label_col: str, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separa una fracción aproximada ``holdout_frac`` de *grupos* del resto.

    Truco: con ``n_splits = round(1 / holdout_frac)`` cada fold de
    StratifiedGroupKFold contiene ~holdout_frac de los grupos. Nos quedamos con
    el primero como holdout. Es aproximado porque los pacientes no se pueden
    partir (uno con 40 cortes pesa más que uno con 2).
    """
    n_splits = max(2, round(1 / holdout_frac))
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    rest_idx, holdout_idx = next(sgkf.split(df, y=df[label_col], groups=df[group_col]))
    return df.iloc[rest_idx], df.iloc[holdout_idx]


def patient_level_split(
    df: pd.DataFrame,
    val_frac: float = 0.15,
    test_frac: float = 0.20,
    group_col: str = "patient_id",
    label_col: str = "label",
    seed: int = 42,
) -> pd.DataFrame:
    """Añade una columna ``split`` ∈ {train, val, test} sin mezclar pacientes.

    Parameters
    ----------
    df : manifest con al menos ``group_col`` y ``label_col``.
    val_frac, test_frac : fracciones aproximadas *de pacientes*.
    seed : semilla; mismo seed + mismo manifest = mismo split siempre.

    Returns
    -------
    Copia del manifest con la columna ``split`` añadida.
    """
    if not 0 < val_frac < 1 or not 0 < test_frac < 1 or val_frac + test_frac >= 1:
        raise ValueError("val_frac y test_frac deben estar en (0, 1) y sumar < 1")

    df = df.reset_index(drop=True)
    # 1) Separamos test del total
    trainval, test = _group_fold_split(df, test_frac, group_col, label_col, seed)
    # 2) Separamos val de lo que queda (re-escalando la fracción)
    train, val = _group_fold_split(trainval, val_frac / (1 - test_frac), group_col, label_col, seed)

    out = df.copy()
    out.loc[train.index, "split"] = "train"
    out.loc[val.index, "split"] = "val"
    out.loc[test.index, "split"] = "test"
    assert_no_patient_leakage(out, group_col)
    return out


def assert_no_patient_leakage(df: pd.DataFrame, group_col: str = "patient_id") -> None:
    """Lanza ``ValueError`` si algún paciente aparece en más de un split.

    Se llama automáticamente tras el split y también antes de entrenar, así que
    un manifest construido a mano con fuga no puede colarse sin aviso.
    """
    splits_per_patient = df.groupby(group_col)["split"].nunique()
    leaked = splits_per_patient[splits_per_patient > 1]
    if not leaked.empty:
        examples = ", ".join(map(str, leaked.index[:5]))
        raise ValueError(
            f"Fuga de datos: {len(leaked)} pacientes aparecen en varios splits (p. ej. {examples})"
        )


def split_summary(df: pd.DataFrame, group_col: str = "patient_id") -> pd.DataFrame:
    """Tabla resumen: nº de pacientes e imágenes por split y clase (para el README/log)."""
    return (
        df.groupby(["split", "label"])
        .agg(pacientes=(group_col, "nunique"), imagenes=("image_path", "count"))
        .reindex(SPLITS, level=0)
    )
