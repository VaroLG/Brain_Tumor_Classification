"""Estadística del análisis radiogenómico.

La pieza central es la **correlación parcial de Spearman**: ¿se asocian la
necrosis y la hipoxia *una vez descontado* lo que explican la edad y el tamaño
del tumor?

Cómo se calcula (y por qué así)
-------------------------------
1. Rangos: Spearman = Pearson sobre rangos. Usar rangos hace el análisis
   robusto a valores extremos y a relaciones monótonas no lineales (la fracción
   necrótica está acotada en [0, 1] y suele estar sesgada).
2. Residualizar: se regresan los rangos de X y de Y sobre los rangos de las
   covariables y nos quedamos con los residuos, es decir, la parte de X y de Y
   que las covariables NO explican.
3. Correlación de Pearson entre ambos residuos = ρ parcial.

Inferencia
----------
- p por permutación: se baraja el residuo de X muchas veces para construir la
  distribución de ρ cuando no hay asociación. No asume normalidad.
- IC por bootstrap: se remuestrean PACIENTES con reemplazo y se repite todo el
  cálculo (incluida la residualización) para ver cuánto varía ρ.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy import stats


# ----------------------------------------------------------------------------
# Utilidades de álgebra
# ----------------------------------------------------------------------------
def _design(covars: np.ndarray | None, n: int) -> np.ndarray:
    """Matriz de diseño con intercepto (n × (k+1))."""
    ones = np.ones((n, 1))
    if covars is None or covars.size == 0:
        return ones
    return np.column_stack([ones, covars])


def _residualize(y: np.ndarray, design: np.ndarray) -> np.ndarray:
    """Residuos de la regresión MCO de ``y`` (vector o matriz por columnas) sobre ``design``."""
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    return y - design @ beta


def _rank(a: np.ndarray) -> np.ndarray:
    """Rangos por columna (empates → rango medio)."""
    return stats.rankdata(a, axis=0)


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a - a.mean(), b - b.mean()
    denom = np.sqrt((a**2).sum() * (b**2).sum())
    return float((a * b).sum() / denom) if denom > 0 else float("nan")


def _p_from_r(r: float | np.ndarray, n: int, k: int) -> np.ndarray:
    """p bilateral analítico para una correlación (parcial) con ``k`` covariables.

    t = r·sqrt(df / (1-r²)), con df = n - 2 - k grados de libertad.
    Se usa en el análisis exploratorio (miles de tests, permutar sería lento);
    en el confirmatorio se usa permutación.
    """
    df = n - 2 - k
    r = np.clip(np.asarray(r, dtype=float), -0.999999, 0.999999)
    t = r * np.sqrt(df / (1 - r**2))
    return 2 * stats.t.sf(np.abs(t), df)


# ----------------------------------------------------------------------------
# Correlación parcial de Spearman
# ----------------------------------------------------------------------------
@dataclass
class PartialResult:
    rho: float
    p_analytic: float
    p_perm: float | None
    ci95: tuple[float, float] | None
    n: int
    covariates: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


def _complete_cases(x: pd.Series, y: pd.Series, covars: pd.DataFrame | None) -> pd.DataFrame:
    parts = [x.rename("x"), y.rename("y")]
    if covars is not None and not covars.empty:
        parts.append(covars)
    return pd.concat(parts, axis=1, join="inner").dropna()


def partial_spearman_arrays(
    x: np.ndarray, y: np.ndarray, z: np.ndarray | None
) -> tuple[float, np.ndarray, np.ndarray]:
    """ρ parcial de Spearman sobre arrays alineados. Devuelve (ρ, residuo_x, residuo_y)."""
    n = len(x)
    design = _design(None if z is None else _rank(z), n)
    rx = _residualize(_rank(x), design)
    ry = _residualize(_rank(y), design)
    return _pearson(rx, ry), rx, ry


def partial_spearman(
    x: pd.Series,
    y: pd.Series,
    covars: pd.DataFrame | None = None,
    n_perm: int = 10_000,
    n_boot: int = 5_000,
    seed: int = 42,
) -> PartialResult:
    """ρ parcial de Spearman entre ``x`` e ``y`` ajustando por ``covars``.

    Las tres entradas se alinean por índice (paciente) y se usan solo los
    pacientes con todos los datos (casos completos).
    """
    d = _complete_cases(x, y, covars)
    cov_cols = [c for c in d.columns if c not in ("x", "y")]
    z = d[cov_cols].to_numpy(float) if cov_cols else None
    xv, yv = d["x"].to_numpy(float), d["y"].to_numpy(float)
    n = len(d)
    if n < len(cov_cols) + 5:
        raise ValueError(f"Muy pocos pacientes ({n}) para {len(cov_cols)} covariables")
    rho, rx, ry = partial_spearman_arrays(xv, yv, z)
    rng = np.random.default_rng(seed)

    p_perm = None
    if n_perm:
        # Permutación vectorizada: cada fila de `perm` es un barajado del residuo de x.
        perm = np.argsort(rng.random((n_perm, n)), axis=1)
        rxp = rx[perm]
        rxp = rxp - rxp.mean(axis=1, keepdims=True)
        ryc = ry - ry.mean()
        null = (rxp @ ryc) / np.sqrt((rxp**2).sum(axis=1) * (ryc**2).sum())
        # +1 en numerador y denominador: el valor observado cuenta como una permutación
        p_perm = float((np.sum(np.abs(null) >= abs(rho)) + 1) / (n_perm + 1))

    ci = None
    if n_boot:
        boots = []
        for _ in range(n_boot):
            idx = rng.integers(0, n, n)
            if len(np.unique(xv[idx])) < 3 or len(np.unique(yv[idx])) < 3:
                continue
            r_b, _, _ = partial_spearman_arrays(xv[idx], yv[idx], None if z is None else z[idx])
            boots.append(r_b)
        ci = (float(np.nanquantile(boots, 0.025)), float(np.nanquantile(boots, 0.975)))

    return PartialResult(
        float(rho), float(_p_from_r(rho, n, len(cov_cols))), p_perm, ci, n, cov_cols
    )


# ----------------------------------------------------------------------------
# Control negativo: firmas aleatorias del mismo tamaño
# ----------------------------------------------------------------------------
def random_signature_null(
    x: pd.Series,
    data: pd.DataFrame,
    signature_size: int,
    covars: pd.DataFrame | None = None,
    n_sets: int = 1_000,
    exclude: list[str] | None = None,
    seed: int = 42,
) -> np.ndarray:
    """Distribución de ρ parcial para firmas ALEATORIAS de ``signature_size`` genes.

    Cada firma aleatoria se puntúa con media de z-scores (mismo método que la
    primaria). Responde a: ¿la hipoxia se asocia a la necrosis MÁS que un
    conjunto cualquiera de genes? En tumores la expresión está muy
    correlacionada (pureza, proliferación...), así que la distribución nula NO
    está centrada en cero necesariamente; por eso este control es informativo.
    """
    common = data.index.intersection(x.dropna().index)
    if covars is not None:
        common = common.intersection(covars.dropna().index)
    d = data.loc[common]
    pool = [g for g in d.columns if g not in set(exclude or [])]
    zmat = ((d[pool] - d[pool].mean()) / d[pool].std(ddof=1).replace(0, np.nan)).to_numpy(float)
    xv = x.loc[common].to_numpy(float)
    zc = None if covars is None else covars.loc[common].to_numpy(float)
    design = _design(None if zc is None else _rank(zc), len(common))
    rx = _residualize(_rank(xv), design)

    rng = np.random.default_rng(seed)
    scores = np.empty((len(common), n_sets))
    for j in range(n_sets):
        cols = rng.choice(len(pool), size=signature_size, replace=False)
        scores[:, j] = np.nanmean(zmat[:, cols], axis=1)
    ry = _residualize(_rank(scores), design)  # residualiza todas las firmas a la vez
    rxc = rx - rx.mean()
    ryc = ry - ry.mean(axis=0)
    return (rxc @ ryc) / np.sqrt((rxc**2).sum() * (ryc**2).sum(axis=0))


def empirical_percentile(observed: float, null: np.ndarray) -> float:
    """Proporción de firmas aleatorias con |ρ| menor que el observado (0–1)."""
    return float(np.mean(np.abs(null) < abs(observed)))


# ----------------------------------------------------------------------------
# Comparar y combinar cohortes (z de Fisher)
# ----------------------------------------------------------------------------
def _fisher(rho: float, n: int, k: int) -> tuple[float, float]:
    """z de Fisher y su error estándar para una correlación parcial con k covariables."""
    return float(np.arctanh(np.clip(rho, -0.999999, 0.999999))), 1 / np.sqrt(n - 3 - k)


def compare_and_pool(rho_a: float, n_a: int, rho_b: float, n_b: int, k: int) -> dict:
    """Heterogeneidad entre dos cohortes y estimación combinada (efectos fijos).

    - Heterogeneidad: z = (z_a - z_b) / sqrt(se_a² + se_b²). Si p ≥ 0.05, las dos
      estimaciones son compatibles con un mismo efecto verdadero.
    - Combinada: media de las z ponderada por 1/se² (= n - 3 - k), devuelta a la
      escala de ρ con tanh. Es la forma estándar de un metaanálisis de
      correlaciones con dos estudios.
    """
    za, sa = _fisher(rho_a, n_a, k)
    zb, sb = _fisher(rho_b, n_b, k)
    z_diff = (za - zb) / np.sqrt(sa**2 + sb**2)
    wa, wb = 1 / sa**2, 1 / sb**2
    zp = (wa * za + wb * zb) / (wa + wb)
    sp = 1 / np.sqrt(wa + wb)
    return {
        "heterogeneity_p": float(2 * stats.norm.sf(abs(z_diff))),
        "pooled_rho": float(np.tanh(zp)),
        "pooled_ci95": (float(np.tanh(zp - 1.96 * sp)), float(np.tanh(zp + 1.96 * sp))),
        "pooled_p": float(2 * stats.norm.sf(abs(zp / sp))),
    }


def validation_verdict(rho_val: float, p_one_sided: float, heterogeneity_p: float) -> str:
    """Clasificación en tres niveles definida en el plan (sección 4.3)."""
    if rho_val > 0 and p_one_sided < 0.05:
        return "replica"
    if rho_val > 0 and heterogeneity_p >= 0.05:
        return "consistente, potencia insuficiente"
    return "no replica"


# ----------------------------------------------------------------------------
# Análisis exploratorio: una característica tras otra
# ----------------------------------------------------------------------------
def bh_fdr(p: np.ndarray) -> np.ndarray:
    """q-valores de Benjamini-Hochberg.

    Con miles de tests, un p < 0.05 aparece por azar en el 5 % de ellos. BH
    controla la proporción esperada de falsos positivos ENTRE los resultados
    declarados significativos (FDR).
    """
    p = np.asarray(p, dtype=float)
    q = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    pv = p[ok]
    m = len(pv)
    if m == 0:
        return q
    order = np.argsort(pv)
    ranked = pv[order] * m / np.arange(1, m + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]  # monotonía
    out = np.empty(m)
    out[order] = np.minimum(ranked, 1.0)
    q[ok] = out
    return q


def layer_association(
    x: pd.Series,
    layer: pd.DataFrame,
    covars: pd.DataFrame | None = None,
    max_missing: float = 0.2,
    min_n: int = 20,
) -> pd.DataFrame:
    """ρ parcial de Spearman de ``x`` contra CADA columna de ``layer``, con FDR.

    Sirve para cualquier capa ómica (genes, proteínas, fosfositios, miRNAs,
    tipos celulares...). Las características con más de ``max_missing`` de
    valores ausentes se descartan; las que tienen algunos ausentes se calculan
    con sus casos completos.
    """
    idx = layer.index.intersection(x.dropna().index)
    if covars is not None:
        idx = idx.intersection(covars.dropna().index)
    lay = layer.loc[idx]
    lay = lay.loc[:, lay.isna().mean() <= max_missing]
    lay = lay.loc[:, lay.nunique() > 2]  # sin variación no hay correlación posible
    xv = x.loc[idx].to_numpy(float)
    zv = None if covars is None else covars.loc[idx].to_numpy(float)
    k = 0 if zv is None else zv.shape[1]

    rhos, ns = {}, {}
    complete = lay.columns[lay.notna().all()]
    if len(complete):
        design = _design(None if zv is None else _rank(zv), len(idx))
        rx = _residualize(_rank(xv), design)
        ry = _residualize(_rank(lay[complete].to_numpy(float)), design)
        rxc, ryc = rx - rx.mean(), ry - ry.mean(axis=0)
        r = (rxc @ ryc) / np.sqrt((rxc**2).sum() * (ryc**2).sum(axis=0))
        rhos.update(dict(zip(complete, r, strict=True)))
        ns.update(dict.fromkeys(complete, len(idx)))
    for col in lay.columns.difference(complete):
        m = lay[col].notna().to_numpy()
        if m.sum() < max(min_n, k + 5):
            continue
        r, _, _ = partial_spearman_arrays(
            xv[m], lay[col].to_numpy(float)[m], None if zv is None else zv[m]
        )
        rhos[col], ns[col] = r, int(m.sum())

    out = pd.DataFrame({"rho": pd.Series(rhos), "n": pd.Series(ns)})
    out["p"] = [float(_p_from_r(r, n, k)) for r, n in zip(out["rho"], out["n"], strict=True)]
    out["q"] = bh_fdr(out["p"].to_numpy())
    return out.sort_values("p")
