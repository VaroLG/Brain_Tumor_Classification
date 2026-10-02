"""Demo del estudio radiogenómico con cohortes sintéticas (sin descargar nada).

Genera dos cohortes con la misma estructura que TCGA y CPTAC, escribe los CSV,
el GMT y el YAML exactamente como lo harían los comandos ``btc rg-*``, y
ejecuta ``btc rg-analyze``. Lo hace dos veces:

    efecto plantado moderado (ρ real ≈ 0.3)  -> el estudio debería detectarlo
    sin efecto                                -> el estudio NO debería detectar nada

    python scripts/demo_radiogenomics.py [--out runs/demo_rg]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from btc.cli import main as btc_main
from btc.radiogenomics.signatures import write_gmt
from btc.radiogenomics.synthetic import make_cohort


def write_scenario(root: Path, effect: float) -> Path:
    disc = make_cohort(100, effect=effect, seed=11, prefix="TCGA")
    val = make_cohort(66, effect=effect, seed=12, prefix="CPTAC")
    data = root / "data"
    data.mkdir(parents=True, exist_ok=True)
    write_gmt(disc["signatures"], data / "firmas.gmt")

    def dump(c: dict, name: str) -> dict:
        c["imaging"].to_csv(data / f"{name}_imaging.csv")
        c["clinical"].to_csv(data / f"{name}_clinical.csv")
        layers = {}
        for k, df in c["layers"].items():
            df.to_csv(data / f"{name}_{k}.csv")
            layers[k] = str(data / f"{name}_{k}.csv")
        return {
            "name": name.upper(),
            "imaging": str(data / f"{name}_imaging.csv"),
            "clinical": str(data / f"{name}_clinical.csv"),
            "layers": layers,
        }

    d_spec = dump(disc, "tcga")
    d_spec["layers"] = {"rna": d_spec["layers"]["rna"]}  # TCGA: solo microarray de ARN
    cfg = {
        "data": {
            "gmt": [str(data / "firmas.gmt")],
            "discovery": d_spec,
            "validation": dump(val, "cptac"),
        },
        "analysis": {
            "primary_signature": "HYPOXIA_SYNTH",
            "sensitivity_signatures": ["HYPOXIA_ALT"],
            "n_perm": 5000,
            "n_boot": 2000,
            "n_random_sets": 500,
        },
    }
    path = root / "estudio.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/demo_rg")
    args = ap.parse_args()
    for label, effect in [("con_efecto", 0.4), ("sin_efecto", 0.0)]:
        root = Path(args.out) / label
        print(f"\n=== Escenario: {label} (efecto = {effect}) ===")
        cfg = write_scenario(root, effect)
        btc_main(["rg-analyze", "--config", str(cfg), "--out-dir", str(root / "resultados")])
        print(f"Resumen: {root / 'resultados' / 'resumen.md'}")


if __name__ == "__main__":
    main()
