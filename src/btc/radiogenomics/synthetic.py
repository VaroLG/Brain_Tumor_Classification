"""Cohorte multiómica sintética con efecto conocido (para tests y demo).

Por qué hace falta
------------------
Antes de aplicar un método estadístico a datos reales hay que comprobar dos
cosas con datos donde conocemos la verdad:

1. **Sensibilidad:** si plantamos una asociación necrosis–hipoxia, ¿el pipeline
   la recupera, con el signo correcto y un IC que contiene el valor real?
2. **Especificidad:** si NO hay asociación (``effect=0``), ¿el pipeline se
   abstiene, o se inventa un resultado?

Cómo se genera (modelo generativo simplificado)
-----------------------------------------------
- Cada paciente tiene una **hipoxia latente** h ~ N(0, 1).
- Imagen: la fracción necrótica depende de h con fuerza ``effect`` (vía una
  logística, para que quede en [0, 1]) y el volumen total depende algo de h
  (más hipoxia → tumores algo mayores), creando el factor de confusión que
  la covariable debe absorber.
- Transcriptoma: ``n_genes`` genes con 3 factores latentes compartidos (como en
  un tumor real, donde pureza, proliferación, inmunidad mueven miles de genes a
  la vez). Los genes de la firma de hipoxia cargan además sobre h.
- Proteoma: solo se mide un ``protein_coverage`` de los genes, y cada proteína
  correlaciona imperfectamente con su ARN (como en la realidad, r ≈ 0.4-0.6).
- Otras capas: fosfositios, fracciones celulares (macrófagos ↑ con h) y pureza
  tumoral (↓ con la necrosis).

No pretende ser realista; pretende tener la ESTRUCTURA de los datos reales.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_cohort(
    n_patients: int = 100,
    effect: float = 0.8,
    n_genes: int = 1500,
    signature_size: int = 100,
    protein_coverage: float = 0.5,
    prefix: str = "SYN",
    seed: int = 0,
) -> dict:
    """Devuelve un dict con ``imaging``, ``clinical``, ``layers`` y ``signatures``."""
    rng = np.random.default_rng(seed)
    pids = [f"{prefix}-{i:04d}" for i in range(n_patients)]
    h = rng.normal(size=n_patients)

    # --- Imagen -------------------------------------------------------------
    v_total = np.exp(10.5 + 0.3 * h + rng.normal(0, 0.4, n_patients))  # mm³, lognormal
    logit_nf = -0.8 + effect * h + rng.normal(0, 0.8, n_patients)
    nf_core = 1 / (1 + np.exp(-logit_nf))
    imaging = pd.DataFrame(
        {"necrosis_fraction_core": nf_core, "v_total": v_total, "log_v_total": np.log1p(v_total)},
        index=pd.Index(pids, name="patient_id"),
    )
    imaging["log_v_ncr"] = np.log1p(nf_core * 0.4 * v_total)

    # --- Clínica ------------------------------------------------------------
    clinical = pd.DataFrame(
        {
            "age": rng.normal(60, 10, n_patients).round(),
            "idh_status": rng.choice(
                ["wildtype", "mutant", None], n_patients, p=[0.85, 0.05, 0.10]
            ),
        },
        index=imaging.index,
    )

    # --- Transcriptoma --------------------------------------------------------
    genes = [f"G{i:05d}" for i in range(n_genes)]
    sig = genes[:signature_size]
    alt_sig = genes[signature_size // 2 : signature_size // 2 + 40]  # firma alternativa solapada
    factors = rng.normal(size=(n_patients, 3))
    loadings = rng.normal(0, 0.5, size=(3, n_genes))
    rna = 8 + factors @ loadings + rng.normal(0, 1, (n_patients, n_genes))
    hyp_load = np.zeros(n_genes)
    hyp_load[: signature_size // 2 + 40] = rng.uniform(0.4, 1.0, signature_size // 2 + 40)
    rna += np.outer(h, hyp_load)
    rna_df = pd.DataFrame(rna, index=imaging.index, columns=genes)

    # --- Proteoma: cobertura parcial y correlación imperfecta con el ARN ------
    measured = rng.choice(n_genes, int(n_genes * protein_coverage), replace=False)
    prot = 0.6 * (rna[:, measured] - rna[:, measured].mean(0)) + rng.normal(
        0, 1, (n_patients, len(measured))
    )
    prot[rng.random(prot.shape) < 0.05] = np.nan  # valores ausentes típicos de proteómica
    prot_df = pd.DataFrame(prot, index=imaging.index, columns=[genes[i] for i in measured])

    # --- Otras capas ----------------------------------------------------------
    phospho = pd.DataFrame(
        rng.normal(size=(n_patients, 200)),
        index=imaging.index,
        columns=[f"{genes[i]}|S{i}" for i in range(200)],
    )
    phospho.iloc[:, :10] += 0.6 * h[:, None]  # 10 fosfositios responden a la hipoxia
    cells = pd.DataFrame(
        {
            "Macrophages": 0.5 * h + rng.normal(0, 1, n_patients),
            "T_cells": rng.normal(0, 1, n_patients),
            "Endothelial": 0.3 * h + rng.normal(0, 1, n_patients),
        },
        index=imaging.index,
    )
    purity = pd.DataFrame(
        {"purity": np.clip(0.8 - 0.3 * nf_core + rng.normal(0, 0.05, n_patients), 0, 1)},
        index=imaging.index,
    )

    return {
        "imaging": imaging,
        "clinical": clinical,
        "layers": {
            "rna": rna_df,
            "protein": prot_df,
            "phospho": phospho,
            "cells": cells,
            "purity": purity,
        },
        "signatures": {"HYPOXIA_SYNTH": sig, "HYPOXIA_ALT": alt_sig},
        "truth": {"hypoxia": pd.Series(h, index=imaging.index), "effect": effect},
    }
