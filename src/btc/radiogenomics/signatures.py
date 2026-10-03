"""Firmas génicas: lectura de ficheros GMT y puntuación por muestra.

Qué es una "puntuación de firma"
--------------------------------
Una firma es una lista de genes que se activan juntos en un proceso biológico
(p. ej. los ~200 genes de HALLMARK_HYPOXIA). Para cada paciente queremos UN
número que resuma "cuánto está activo ese programa". Dos métodos, ambos por
muestra (no dependen de comparar grupos):

1. **Media de z-scores** (método primario del plan). Cada gen se estandariza
   en la cohorte (media 0, desviación 1) y se promedian los genes de la firma.
   Ventaja: transparente. Inconveniente: depende de la cohorte (un z-score es
   relativo a los demás pacientes), por eso se calcula por separado en TCGA y
   en CPTAC y nunca se mezclan cohortes antes de estandarizar.

2. **Singscore** (sensibilidad). Dentro de cada paciente se ordenan todos los
   genes por expresión; la puntuación es el rango medio de los genes de la
   firma, normalizado a [0, 1]. Ventaja: solo usa información de ese paciente,
   robusto a diferencias de escala entre plataformas (microarray vs RNA-seq vs
   proteína).

Formato GMT (MSigDB)
--------------------
Una línea por firma, separada por tabuladores:
``NOMBRE<TAB>descripción_o_URL<TAB>GEN1<TAB>GEN2...``
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


def read_gmt(path: str | Path) -> dict[str, list[str]]:
    """Lee un GMT y devuelve ``{nombre_firma: [genes]}`` (genes en mayúsculas, sin duplicados)."""
    sets: dict[str, list[str]] = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            genes = [g.strip().upper() for g in parts[2:] if g.strip()]
            sets[parts[0]] = list(dict.fromkeys(genes))  # conserva orden, quita duplicados
    if not sets:
        raise ValueError(f"No se encontró ninguna firma en {path}")
    return sets


def write_gmt(sets: dict[str, list[str]], path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for name, genes in sets.items():
            fh.write("\t".join([name, "na", *genes]) + "\n")


@dataclass
class Coverage:
    n_signature: int
    n_measured: int
    measured: list[str]

    @property
    def fraction(self) -> float:
        return self.n_measured / self.n_signature if self.n_signature else 0.0


def signature_coverage(data: pd.DataFrame, genes: list[str]) -> Coverage:
    """Cuántos genes de la firma se midieron en esta plataforma.

    Es crítico reportarlo: en proteómica suele cuantificarse solo una parte de
    la firma, y una puntuación con 30 de 200 genes no es comparable a otra con
    190 sin decirlo.
    """
    cols = {c.upper(): c for c in data.columns}
    measured = [cols[g] for g in genes if g in cols]
    return Coverage(len(genes), len(measured), measured)


def score_mean_z(data: pd.DataFrame, genes: list[str], min_genes: int = 10) -> pd.Series:
    """Media de z-scores de los genes de la firma (método primario).

    Se ignoran los valores ausentes (frecuentes en proteómica) promediando solo
    los genes medidos en cada paciente.
    """
    cov = signature_coverage(data, genes)
    if cov.n_measured < min_genes:
        raise ValueError(f"Solo {cov.n_measured} genes de la firma medidos (mínimo {min_genes})")
    sub = data[cov.measured].astype(float)
    sd = sub.std(ddof=1).replace(0, np.nan)  # un gen constante no aporta información
    z = (sub - sub.mean()) / sd
    return z.mean(axis=1, skipna=True).rename("score")


def score_singscore(data: pd.DataFrame, genes: list[str], min_genes: int = 10) -> pd.Series:
    """Singscore simplificado (Foroutan et al., 2018) para una firma "hacia arriba".

    Para cada paciente: rango de cada gen entre TODOS los genes medidos (1 = el
    menos expresado), media de los rangos de la firma, y normalización lineal
    entre el mínimo y el máximo teóricos para ese tamaño de firma → [0, 1].
    """
    cov = signature_coverage(data, genes)
    if cov.n_measured < min_genes:
        raise ValueError(f"Solo {cov.n_measured} genes de la firma medidos (mínimo {min_genes})")
    ranks = data.astype(float).rank(axis=1)  # rangos dentro de cada paciente
    n_genes = data.notna().sum(axis=1)
    k = ranks[cov.measured].notna().sum(axis=1)
    mean_rank = ranks[cov.measured].mean(axis=1)
    low = (k + 1) / 2  # media de rangos si la firma fueran los k menores
    high = n_genes - (k - 1) / 2  # media si fueran los k mayores
    return ((mean_rank - low) / (high - low)).rename("score")


SCORERS = {"mean_z": score_mean_z, "singscore": score_singscore}
