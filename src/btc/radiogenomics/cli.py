"""Comandos ``btc rg-*`` del estudio radiogenómico.

Orden de uso (ver docs/PLAN_ANALISIS_RADIOGENOMICA.md):

    1. btc rg-imaging     máscaras BraTS          -> imaging.csv
    2. btc rg-overlap     imaging.csv + lista IDs -> decisión de factibilidad
    3. btc rg-fetch-tcga  cBioPortal              -> expresión + clínica (TCGA)
       btc rg-fetch-cptac paquete cptac           -> capas ómicas + clínica (CPTAC)
    4. btc rg-analyze     YAML del estudio        -> informe.json, resumen.md, figuras
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from btc.utils import get_logger

log = get_logger("btc.rg")


def _read_table(path: str | Path) -> pd.DataFrame:
    """CSV con ``patient_id`` como primera columna → DataFrame indexado por paciente."""
    df = pd.read_csv(path, index_col=0)
    df.index = df.index.astype(str)
    df.index.name = "patient_id"
    return df


def _cmd_imaging(a: argparse.Namespace) -> None:
    from btc.radiogenomics.imaging import imaging_table

    df = imaging_table(a.seg_root, a.pattern, a.cohort, a.id_from)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out)
    log.info(
        "%d pacientes. Fracción necrótica (núcleo): mediana %.2f, rango [%.2f, %.2f]",
        len(df),
        df.necrosis_fraction_core.median(),
        df.necrosis_fraction_core.min(),
        df.necrosis_fraction_core.max(),
    )


def _cmd_overlap(a: argparse.Namespace) -> None:
    from btc.radiogenomics.cohorts import CBioPortal, normalize_patient_id, overlap_report

    img_ids = _read_table(a.imaging).index
    if a.cbioportal_list:
        raw = CBioPortal(a.study).sample_ids(a.cbioportal_list)
    else:
        raw = Path(a.omics_ids).read_text().split()
    omics = {normalize_patient_id(s, a.cohort) for s in raw}
    rep = overlap_report(img_ids, omics)
    print(rep)
    if rep.only_imaging:
        print(
            f"\nCon imagen pero sin ómica ({len(rep.only_imaging)}): "
            f"{', '.join(rep.only_imaging[:15])}{' ...' if len(rep.only_imaging) > 15 else ''}"
        )


def _cmd_fetch_tcga(a: argparse.Namespace) -> None:
    """Descarga expresión (genes de las firmas) + clínica + IDH desde cBioPortal."""
    from btc.radiogenomics.cohorts import CBioPortal, ensure_log_scale
    from btc.radiogenomics.signatures import read_gmt

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    genes: set[str] = set()
    for g in a.gmt:
        for genes_set in read_gmt(g).values():
            genes.update(genes_set)
    if a.genes_file:
        genes.update(Path(a.genes_file).read_text().split())
    log.info("Descargando %d genes (unión de las firmas = universo del análisis)", len(genes))

    cb = CBioPortal(a.study)
    expr = ensure_log_scale(cb.expression(sorted(genes), a.profile, a.sample_list))
    expr.to_csv(out / "tcga_expression.csv")

    clin = cb.clinical()
    if a.age_attr not in clin:
        raise SystemExit(
            f"Atributo de edad '{a.age_attr}' no encontrado. Disponibles: {list(clin.columns)}"
        )
    clinical = pd.DataFrame({"age": pd.to_numeric(clin[a.age_attr], errors="coerce")})
    for col in ("OS_MONTHS", "OS_STATUS", "SEX"):
        if col in clin:
            clinical[col.lower()] = clin[col]
    clinical = clinical.join(cb.idh_status(), how="left")  # ausente = IDH desconocido
    clinical.to_csv(out / "tcga_clinical.csv")
    log.info(
        "Expresión: %d pacientes × %d genes; clínica: %d pacientes (IDH conocido en %d)",
        *expr.shape,
        len(clinical),
        clinical["idh_status"].notna().sum(),
    )


def _cmd_fetch_cptac(a: argparse.Namespace) -> None:
    """Descarga cada capa ómica de CPTAC-GBM con el paquete ``cptac``."""
    from btc.radiogenomics.cohorts import CPTAC_LAYERS, load_cptac_layer, normalize_patient_id

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for layer in a.layers:
        try:
            df = load_cptac_layer(layer)
        except Exception as err:  # una capa que falla no debe tumbar las demás
            log.warning("Capa %s no descargada: %s", layer, err)
            continue
        df.to_csv(out / f"cptac_{layer}.csv")
        log.info("%s: %d pacientes × %d características", layer, *df.shape)

    import cptac

    gbm = cptac.Gbm()
    clin = gbm.get_clinical(source="mssm", tissue_type="tumor")
    if a.age_col not in clin:
        raise SystemExit(
            f"Columna de edad '{a.age_col}' no encontrada. Disponibles: {list(clin.columns)}"
        )
    clinical = pd.DataFrame({"age": pd.to_numeric(clin[a.age_col], errors="coerce")})
    clinical.index = [normalize_patient_id(i, "cptac") for i in clinical.index]
    clinical = clinical.groupby(level=0).first()
    try:  # estado IDH a partir de las mutaciones somáticas
        muts = gbm.get_somatic_mutation(source="washu")
        genes_col = "Gene" if "Gene" in muts else muts.columns[0]
        mutated = {
            normalize_patient_id(i, "cptac")
            for i, g in zip(muts.index, muts[genes_col], strict=True)
            if g in ("IDH1", "IDH2")
        }
        sequenced = {normalize_patient_id(i, "cptac") for i in muts.index}
        clinical["idh_status"] = [
            "mutant" if p in mutated else ("wildtype" if p in sequenced else None)
            for p in clinical.index
        ]
    except Exception as err:
        log.warning("Sin estado IDH: %s", err)
    clinical.index.name = "patient_id"
    clinical.to_csv(out / "cptac_clinical.csv")
    log.info(
        "Clínica: %d pacientes. Capas disponibles en cptac: %s", len(clinical), list(CPTAC_LAYERS)
    )


def _load_cohort(spec: dict):
    from btc.radiogenomics.analysis import CohortData

    license_ = spec.get("license", "open")
    if license_ not in ("open", "restricted"):
        raise SystemExit(f"license debe ser 'open' o 'restricted' (cohorte {spec['name']})")
    return CohortData(
        name=spec["name"],
        imaging=_read_table(spec["imaging"]),
        clinical=_read_table(spec["clinical"]),
        layers={k: _read_table(v) for k, v in spec["layers"].items()},
        restricted=license_ == "restricted",
        id_format=spec.get("id_format"),
    )


def _cmd_check_publish(a: argparse.Namespace) -> None:
    """Busca IDs de pacientes de cohortes restringidas fuera de solo_local/ antes de publicar."""
    from btc.radiogenomics.publish import scan_for_patient_ids

    findings = scan_for_patient_ids(a.dir, a.cohorts)
    if not findings:
        print(f"OK: ningún ID de {', '.join(a.cohorts)} fuera de solo_local/ en {a.dir}")
        return
    for f in findings:
        print(f"⚠ {f.path}: {f.n_ids} IDs (p. ej. {f.example})")
    raise SystemExit("Hay ficheros con IDs de pacientes de cohortes restringidas: no publicar")


def _cmd_analyze(a: argparse.Namespace) -> None:
    from btc.radiogenomics.analysis import load_study_config, run_study
    from btc.radiogenomics.signatures import read_gmt

    cfg, data = load_study_config(a.config)
    sigs: dict[str, list[str]] = {}
    for g in data["gmt"]:
        sigs.update(read_gmt(g))
    if cfg.primary_signature not in sigs:
        raise SystemExit(f"La firma primaria {cfg.primary_signature} no está en los GMT indicados")
    disc = _load_cohort(data["discovery"])
    val = _load_cohort(data["validation"]) if data.get("validation") else None
    rep = run_study(disc, sigs, cfg, a.out_dir, validation=val, exploratory=not a.no_exploratory)
    r = rep["discovery"]["confirmatory"]["result"]
    print(
        f"\nConfirmatorio: ρ = {r['rho']:.3f} IC95 {r['ci95']} p = {r['p_perm']:.4f} "
        f"→ {'CUMPLE' if rep['discovery']['confirmatory_success'] else 'NO CUMPLE'} el criterio"
    )


def add_subparsers(sub: argparse._SubParsersAction) -> None:
    s = sub.add_parser("rg-imaging", help="Volúmenes y fracción necrótica desde máscaras BraTS")
    s.add_argument("--seg-root", required=True)
    s.add_argument(
        "--pattern",
        default="**/*_GlistrBoost_ManuallyCorrected.nii.gz",
        help="Glob de las máscaras (BraTS 2021: '**/*_seg.nii.gz')",
    )
    s.add_argument("--cohort", choices=["tcga", "cptac", "other"], default="tcga")
    s.add_argument("--id-from", choices=["prefix", "parent"], default="prefix")
    s.add_argument("--out", required=True)
    s.set_defaults(func=_cmd_imaging)

    s = sub.add_parser("rg-overlap", help="Solapamiento imagen ↔ ómica y potencia (factibilidad)")
    s.add_argument("--imaging", required=True, help="CSV de rg-imaging")
    g = s.add_mutually_exclusive_group(required=True)
    g.add_argument("--cbioportal-list", help="p. ej. gbm_tcga_mrna_U133")
    g.add_argument("--omics-ids", help="Fichero de texto con un ID por línea")
    s.add_argument("--study", default="gbm_tcga")
    s.add_argument("--cohort", choices=["tcga", "cptac"], default="tcga")
    s.set_defaults(func=_cmd_overlap)

    s = sub.add_parser("rg-fetch-tcga", help="Expresión + clínica + IDH de TCGA vía cBioPortal")
    s.add_argument("--gmt", nargs="+", required=True, help="GMT(s) de MSigDB: definen los genes")
    s.add_argument("--genes-file", default=None, help="Genes adicionales (uno por línea)")
    s.add_argument("--out-dir", required=True)
    s.add_argument("--study", default="gbm_tcga")
    s.add_argument("--profile", default="gbm_tcga_mrna_U133")
    s.add_argument("--sample-list", default="gbm_tcga_mrna_U133")
    s.add_argument("--age-attr", default="AGE")
    s.set_defaults(func=_cmd_fetch_tcga)

    s = sub.add_parser("rg-fetch-cptac", help="Capas ómicas + clínica de CPTAC-GBM (paquete cptac)")
    s.add_argument("--out-dir", required=True)
    s.add_argument(
        "--layers",
        nargs="+",
        default=[
            "transcriptomics",
            "proteomics",
            "phosphoproteomics",
            "acetylproteomics",
            "CNV",
            "miRNA",
            "xcell",
            "cibersort",
            "tumor_purity",
        ],
    )
    s.add_argument("--age-col", default="Age")
    s.set_defaults(func=_cmd_fetch_cptac)

    s = sub.add_parser("rg-analyze", help="Ejecuta el estudio completo según un YAML")
    s.add_argument("--config", required=True)
    s.add_argument("--out-dir", required=True)
    s.add_argument("--no-exploratory", action="store_true")
    s.set_defaults(func=_cmd_analyze)

    s = sub.add_parser(
        "rg-check-publish",
        help="Comprueba que no haya IDs de cohortes restringidas fuera de solo_local/",
    )
    s.add_argument("--dir", required=True, help="Carpeta a revisar (p. ej. runs/radiogenomica)")
    s.add_argument("--cohorts", nargs="+", default=["cptac"], choices=["cptac", "tcga", "upenn"])
    s.set_defaults(func=_cmd_check_publish)
