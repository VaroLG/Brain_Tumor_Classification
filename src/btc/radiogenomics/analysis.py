"""Orquestación del estudio según ``docs/PLAN_ANALISIS_RADIOGENOMICA.md``.

Jerarquía de análisis (el orden importa: define qué se puede afirmar)
---------------------------------------------------------------------
1. CONFIRMATORIO  — un único test en la cohorte de descubrimiento (TCGA).
2. ROBUSTEZ       — variantes del confirmatorio; describen, no deciden.
3. VALIDACIÓN     — mismo código en otra cohorte (CPTAC), en ARN y en proteína.
4. EXPLORATORIO   — todas las capas ómicas, característica a característica,
                    con FDR. Genera hipótesis para el futuro.

Cada cohorte se representa con un ``CohortData``: tablas con pacientes en el
índice. El diagrama de flujo (cuántos pacientes quedan tras cada criterio) se
guarda siempre: es lo primero que pide un revisor.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from btc.radiogenomics.signatures import SCORERS, signature_coverage
from btc.radiogenomics.stats import (
    compare_and_pool,
    empirical_percentile,
    layer_association,
    partial_spearman,
    partial_spearman_arrays,
    random_signature_null,
    validation_verdict,
)
from btc.utils import get_logger

log = get_logger(__name__)


@dataclass
class CohortData:
    name: str
    imaging: pd.DataFrame  # índice patient_id; columnas de imaging.region_volumes
    clinical: pd.DataFrame  # índice patient_id; al menos las covariables clínicas
    layers: dict[str, pd.DataFrame]  # capa -> pacientes × características


@dataclass
class StudyConfig:
    primary_signature: str
    sensitivity_signatures: list[str] = field(default_factory=list)
    exposure: str = "necrosis_fraction_core"
    secondary_exposure: str = "log_v_ncr"
    covariates: list[str] = field(default_factory=lambda: ["age", "log_v_total"])
    primary_layer: str = "rna"
    validation_layers: list[str] = field(default_factory=lambda: ["rna", "protein"])
    purity_column: str = "purity"  # covariable extra de sensibilidad si existe
    scorer: str = "mean_z"
    alt_scorer: str = "singscore"
    exclude_idh_mutant: bool = True
    n_perm: int = 10_000
    n_boot: int = 5_000
    n_random_sets: int = 1_000
    fdr: float = 0.10
    min_signature_genes: int = 10
    seed: int = 42

    @classmethod
    def from_dict(cls, d: dict) -> StudyConfig:
        return cls(**d)


# ----------------------------------------------------------------------------
# Preparación de la cohorte
# ----------------------------------------------------------------------------
def prepare_table(
    cohort: CohortData, cfg: StudyConfig, layer: str, drop_idh_unknown: bool = False
) -> tuple[pd.DataFrame, list[tuple[str, int]]]:
    """Une imagen + clínica + disponibilidad de la capa y registra cada exclusión."""
    flow: list[tuple[str, int]] = []
    t = cohort.imaging.copy()
    flow.append(("Con segmentación", len(t)))
    t = t[t[cfg.exposure].notna()]
    flow.append((f"Con {cfg.exposure} calculable", len(t)))
    t = t.join(cohort.clinical, how="inner", rsuffix="_clin")
    flow.append(("Con datos clínicos", len(t)))
    if cfg.exclude_idh_mutant and "idh_status" in t:
        t = t[t["idh_status"] != "mutant"]
        flow.append(("Excluidos IDH-mutantes conocidos", len(t)))
        if drop_idh_unknown:
            t = t[t["idh_status"] == "wildtype"]
            flow.append(("Excluidos IDH desconocido", len(t)))
    t = t[t.index.isin(cohort.layers[layer].index)]
    flow.append((f"Con ómica '{layer}'", len(t)))
    missing_cov = [c for c in cfg.covariates if c not in t.columns]
    if missing_cov:
        raise KeyError(f"Covariables no encontradas en imagen/clínica: {missing_cov}")
    t = t.dropna(subset=cfg.covariates)
    flow.append(("Con todas las covariables", len(t)))
    return t, flow


# ----------------------------------------------------------------------------
# Test de una firma
# ----------------------------------------------------------------------------
def signature_test(
    table: pd.DataFrame,
    data: pd.DataFrame,
    genes: list[str],
    cfg: StudyConfig,
    exposure: str | None = None,
    covariates: list[str] | None = None,
    scorer: str | None = None,
    with_null: bool = False,
) -> dict:
    """ρ parcial entre la exposición y la puntuación de una firma en una capa ómica."""
    exposure = exposure or cfg.exposure
    covariates = cfg.covariates if covariates is None else covariates
    data = data.loc[data.index.intersection(table.index)]
    cov = signature_coverage(data, genes)
    score = SCORERS[scorer or cfg.scorer](data, genes, cfg.min_signature_genes)
    res = partial_spearman(
        table[exposure], score, table[covariates], cfg.n_perm, cfg.n_boot, cfg.seed
    )
    out = {
        "result": res.to_dict(),
        "coverage": {
            "n_signature": cov.n_signature,
            "n_measured": cov.n_measured,
            "fraction": round(cov.fraction, 3),
        },
        "score": score,
    }
    if with_null:
        null = random_signature_null(
            table[exposure],
            data,
            cov.n_measured,
            table[covariates],
            cfg.n_random_sets,
            exclude=cov.measured,
            seed=cfg.seed,
        )
        out["random_null"] = {
            "percentile": empirical_percentile(res.rho, null),
            "null_median": float(np.median(null)),
            "null_q95_abs": float(np.quantile(np.abs(null), 0.95)),
        }
        out["_null"] = null
    return out


def one_sided_p(rho: float, p_two: float) -> float:
    """p unilateral para la hipótesis direccional ρ > 0 a partir del bilateral."""
    return p_two / 2 if rho > 0 else 1 - p_two / 2


# ----------------------------------------------------------------------------
# Fases del estudio
# ----------------------------------------------------------------------------
def run_discovery(cohort: CohortData, sigs: dict[str, list[str]], cfg: StudyConfig) -> dict:
    """Confirmatorio + robustez en la cohorte de descubrimiento."""
    table, flow = prepare_table(cohort, cfg, cfg.primary_layer)
    data = cohort.layers[cfg.primary_layer]
    log.info("[%s] n = %d tras exclusiones", cohort.name, len(table))
    prim = signature_test(table, data, sigs[cfg.primary_signature], cfg, with_null=True)
    r = prim["result"]
    success = bool(r["rho"] > 0 and r["p_perm"] < 0.05)

    robustness = {
        f"scorer_{cfg.alt_scorer}": signature_test(
            table, data, sigs[cfg.primary_signature], cfg, scorer=cfg.alt_scorer
        ),
        f"exposure_{cfg.secondary_exposure}": signature_test(
            table, data, sigs[cfg.primary_signature], cfg, exposure=cfg.secondary_exposure
        ),
    }
    for name in cfg.sensitivity_signatures:
        if name in sigs:
            robustness[f"signature_{name}"] = signature_test(table, data, sigs[name], cfg)
        else:
            log.warning("Firma de sensibilidad '%s' no está en el GMT", name)
    if "idh_status" in table:
        t2, _ = prepare_table(cohort, cfg, cfg.primary_layer, drop_idh_unknown=True)
        if len(t2) >= len(cfg.covariates) + 10:
            robustness["only_idh_wildtype"] = signature_test(
                t2, data, sigs[cfg.primary_signature], cfg
            )
    return {
        "cohort": cohort.name,
        "flow": flow,
        "confirmatory": prim,
        "confirmatory_success": success,
        "robustness": robustness,
        "_table": table,
    }


def run_validation(cohort: CohortData, sigs: dict[str, list[str]], cfg: StudyConfig) -> dict:
    """Validación externa: misma firma, mismos parámetros, en cada capa indicada."""
    out: dict = {"cohort": cohort.name, "layers": {}}
    genes = sigs[cfg.primary_signature]
    for layer in cfg.validation_layers:
        if layer not in cohort.layers:
            log.warning("[%s] capa '%s' no disponible", cohort.name, layer)
            continue
        table, flow = prepare_table(cohort, cfg, layer)
        res = signature_test(table, cohort.layers[layer], genes, cfg)
        r = res["result"]
        p1 = one_sided_p(r["rho"], r["p_perm"])
        entry = {
            "flow": flow,
            "test": res,
            "p_one_sided": p1,
            "replicated": bool(r["rho"] > 0 and p1 < 0.05),
        }
        purity = _find_column(cohort, cfg.purity_column)
        if purity is not None:
            t_p = table.join(purity.rename("purity_cov"), how="inner").dropna(subset=["purity_cov"])
            if len(t_p) >= len(cfg.covariates) + 10:
                entry["with_purity"] = signature_test(
                    t_p,
                    cohort.layers[layer],
                    genes,
                    cfg,
                    covariates=[*cfg.covariates, "purity_cov"],
                )
        out["layers"][layer] = entry
    return out


def _find_column(cohort: CohortData, col: str) -> pd.Series | None:
    """Busca una columna (p. ej. pureza) en la clínica o en cualquier capa."""
    if col in cohort.clinical:
        return cohort.clinical[col]
    for df in cohort.layers.values():
        if col in df:
            return df[col]
    return None


def run_exploratory(cohort: CohortData, cfg: StudyConfig, out_dir: Path) -> dict:
    """Asociación característica a característica en TODAS las capas, con FDR por capa."""
    summary = {}
    out_dir.mkdir(parents=True, exist_ok=True)
    for layer, data in cohort.layers.items():
        table, _ = prepare_table(cohort, cfg, layer)
        if len(table) < len(cfg.covariates) + 10:
            continue
        res = layer_association(table[cfg.exposure], data, table[cfg.covariates])
        path = out_dir / f"exploratorio_{cohort.name}_{layer}.csv"
        res.to_csv(path, index_label="feature")
        hits = res[res["q"] < cfg.fdr]
        summary[layer] = {
            "n_patients": len(table),
            "n_features": len(res),
            f"n_q<{cfg.fdr}": len(hits),
            "top_positive": hits[hits.rho > 0].head(10).index.tolist(),
            "top_negative": hits[hits.rho < 0].head(10).index.tolist(),
            "file": path.name,
        }
    return summary


# ----------------------------------------------------------------------------
# Figuras e informe
# ----------------------------------------------------------------------------
def _plot_residuals(
    table: pd.DataFrame, score: pd.Series, cfg: StudyConfig, rho: float, path: Path, title: str
) -> None:
    """Dispersión de los residuos: la relación que queda tras ajustar por covariables."""
    d = table[[cfg.exposure, *cfg.covariates]].join(score.rename("score"), how="inner").dropna()
    _, rx, ry = partial_spearman_arrays(
        d[cfg.exposure].to_numpy(float),
        d["score"].to_numpy(float),
        d[cfg.covariates].to_numpy(float),
    )
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.scatter(rx, ry, s=18, alpha=0.7, color="#2b6cb0")
    slope, intercept = np.polyfit(rx, ry, 1)
    xs = np.linspace(rx.min(), rx.max(), 50)
    ax.plot(xs, slope * xs + intercept, color="#c53030", lw=1.5)
    ax.set_xlabel(f"{cfg.exposure} (rango residual)")
    ax.set_ylabel(f"{cfg.primary_signature} (rango residual)")
    ax.set_title(f"{title}\nρ parcial = {rho:.2f} (n = {len(d)})", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_null(null: np.ndarray, rho: float, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.hist(null, bins=40, color="#a0aec0")
    ax.axvline(rho, color="#c53030", lw=2, label=f"Hipoxia (ρ = {rho:.2f})")
    ax.set_xlabel("ρ parcial de firmas aleatorias del mismo tamaño")
    ax.set_ylabel("Frecuencia")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _strip_private(obj):
    """Quita Series/arrays/claves privadas para serializar el informe a JSON."""
    if isinstance(obj, dict):
        return {
            k: _strip_private(v)
            for k, v in obj.items()
            if not k.startswith("_")
            and not isinstance(v, (pd.Series, pd.DataFrame))
            and k != "score"
        }
    if isinstance(obj, (list, tuple)):
        return [_strip_private(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def run_study(
    discovery: CohortData,
    sigs: dict[str, list[str]],
    cfg: StudyConfig,
    out_dir: str | Path,
    validation: CohortData | None = None,
    exploratory: bool = True,
) -> dict:
    """Ejecuta el estudio completo y escribe ``informe.json`` + figuras en ``out_dir``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report: dict = {"config": asdict(cfg)}

    disc = run_discovery(discovery, sigs, cfg)
    report["discovery"] = disc
    conf = disc["confirmatory"]
    _plot_residuals(
        disc["_table"],
        conf["score"],
        cfg,
        conf["result"]["rho"],
        out_dir / "confirmatorio_residuos.png",
        f"Descubrimiento ({discovery.name})",
    )
    _plot_null(conf["_null"], conf["result"]["rho"], out_dir / "control_firmas_aleatorias.png")

    if validation is not None:
        val = run_validation(validation, sigs, cfg)
        d = conf["result"]
        for e in val["layers"].values():
            v = e["test"]["result"]
            comp = compare_and_pool(d["rho"], d["n"], v["rho"], v["n"], len(cfg.covariates))
            e["vs_discovery"] = comp
            e["verdict"] = validation_verdict(v["rho"], e["p_one_sided"], comp["heterogeneity_p"])
        report["validation"] = val
    if exploratory:
        report["exploratory"] = {discovery.name: run_exploratory(discovery, cfg, out_dir)}
        if validation is not None:
            report["exploratory"][validation.name] = run_exploratory(validation, cfg, out_dir)

    clean = _strip_private(report)
    (out_dir / "informe.json").write_text(json.dumps(clean, indent=2, ensure_ascii=False))
    (out_dir / "resumen.md").write_text(summary_markdown(clean))
    log.info("Informe en %s", out_dir)
    return report


def summary_markdown(rep: dict) -> str:
    """Resumen legible del informe (lo que iría en el README de resultados)."""
    d = rep["discovery"]
    r = d["confirmatory"]["result"]
    pct = d["confirmatory"]["random_null"]["percentile"]
    verdict = "CUMPLIDO" if d["confirmatory_success"] else "NO CUMPLIDO"
    lines = [
        "# Resumen del estudio radiogenómico",
        "",
        f"## Confirmatorio ({d['cohort']})",
        "",
        "| Paso | n |",
        "|---|---|",
        *[f"| {step} | {n} |" for step, n in d["flow"]],
        "",
        f"- ρ parcial = **{r['rho']:.3f}**, IC 95 % [{r['ci95'][0]:.3f}, {r['ci95'][1]:.3f}], "
        f"p (permutación) = {r['p_perm']:.4f}, n = {r['n']}",
        f"- Cobertura de la firma: {d['confirmatory']['coverage']['n_measured']}"
        f"/{d['confirmatory']['coverage']['n_signature']}",
        f"- Control con firmas aleatorias: percentil {pct:.3f}",
        f"- **Criterio de éxito (ρ > 0 y p < 0.05): {verdict}**",
        "",
        "## Robustez (descriptivo)",
        "",
        "| Variante | ρ | IC 95 % | n |",
        "|---|---|---|---|",
    ]
    for name, t in d["robustness"].items():
        rr = t["result"]
        lines.append(
            f"| {name} | {rr['rho']:.3f} | [{rr['ci95'][0]:.3f}, {rr['ci95'][1]:.3f}] | {rr['n']} |"
        )
    if "validation" in rep:
        v = rep["validation"]
        lines += [
            "",
            f"## Validación externa ({v['cohort']})",
            "",
            "| Capa | ρ | IC 95 % | p unilateral | Cobertura | Veredicto | ρ combinada [IC 95 %] |",
            "|---|---|---|---|---|---|---|",
        ]
        for layer, e in v["layers"].items():
            rr, cov = e["test"]["result"], e["test"]["coverage"]
            pool = e.get("vs_discovery", {})
            pooled = (
                f"{pool['pooled_rho']:.3f} [{pool['pooled_ci95'][0]:.3f}, "
                f"{pool['pooled_ci95'][1]:.3f}]"
                if pool
                else "—"
            )
            lines.append(
                f"| {layer} | {rr['rho']:.3f} | [{rr['ci95'][0]:.3f}, {rr['ci95'][1]:.3f}] | "
                f"{e['p_one_sided']:.4f} | {cov['n_measured']}/{cov['n_signature']} | "
                f"{e.get('verdict', '—')} | {pooled} |"
            )
        lines.append(
            "\n*La ρ combinada de la capa de proteína es orientativa: combina ARN "
            "(descubrimiento) con proteína (validación).*"
        )
    if "exploratory" in rep:
        lines += [
            "",
            "## Exploratorio (genera hipótesis, no conclusiones)",
            "",
            "| Cohorte | Capa | Características | Significativas (FDR) |",
            "|---|---|---|---|",
        ]
        for coh, layers in rep["exploratory"].items():
            for layer, s in layers.items():
                n_hits = next(v for k, v in s.items() if k.startswith("n_q<"))
                lines.append(f"| {coh} | {layer} | {s['n_features']} | {n_hits} |")
    return "\n".join(lines) + "\n"


def load_study_config(path: str | Path) -> tuple[StudyConfig, dict]:
    """Lee el YAML del estudio. Devuelve (config estadística, rutas de datos)."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return StudyConfig.from_dict(raw["analysis"]), raw["data"]
