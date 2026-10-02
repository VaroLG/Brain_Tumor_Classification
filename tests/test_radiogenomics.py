"""Tests del estudio radiogenómico.

Además de comprobar que el código funciona, verifican propiedades estadísticas:
que el método recupera un efecto conocido y que no lo inventa cuando no existe.
"""

import numpy as np
import pandas as pd
import pytest

from btc.radiogenomics.cohorts import (
    CBioPortal,
    flatten_cptac,
    is_primary_tumor_sample,
    min_detectable_r,
    n_for_r,
    normalize_patient_id,
    overlap_report,
)
from btc.radiogenomics.imaging import region_volumes
from btc.radiogenomics.signatures import read_gmt, score_mean_z, score_singscore, write_gmt
from btc.radiogenomics.stats import (
    bh_fdr,
    compare_and_pool,
    layer_association,
    partial_spearman,
    validation_verdict,
)
from btc.radiogenomics.synthetic import make_cohort


# --- identificadores y factibilidad --------------------------------------------
def test_normalize_ids():
    assert normalize_patient_id("TCGA-02-0006-01A-01R-1849-01", "tcga") == "TCGA-02-0006"
    assert normalize_patient_id("tcga-02-0006", "tcga") == "TCGA-02-0006"
    assert normalize_patient_id("C3L-00016.N", "cptac") == "C3L-00016"
    with pytest.raises(ValueError):
        normalize_patient_id("UPENN-GBM-00001", "tcga")


def test_primary_tumor_sample_code():
    assert is_primary_tumor_sample("TCGA-02-0006-01")
    assert not is_primary_tumor_sample("TCGA-02-0006-10")  # sangre normal
    assert not is_primary_tumor_sample("TCGA-02-0006-02")  # recidiva


def test_power_matches_plan_table():
    # Valores de la tabla de factibilidad del plan
    assert round(min_detectable_r(100), 2) == 0.28
    assert round(min_detectable_r(85), 2) == 0.30
    assert round(min_detectable_r(60), 2) == 0.35
    assert n_for_r(0.30) == 85  # inversa coherente


def test_overlap_decision_rule():
    img = [f"P{i}" for i in range(100)]
    assert overlap_report(img, img[:90]).decision == "PROCEDER"
    assert overlap_report(img, img[:70]).decision.startswith("PROCEDER declarando")
    rep = overlap_report(img, img[:40])
    assert rep.decision.startswith("DETENER") and rep.n_overlap == 40
    assert len(rep.only_imaging) == 60


# --- imagen ---------------------------------------------------------------------
def test_region_volumes_and_label_conventions():
    m = np.zeros((10, 10, 10), np.uint8)
    m[:2] = 1  # 200 vóxeles NCR
    m[2:5] = 4  # 300 vóxeles ET (BraTS clásico)
    m[5:9] = 2  # 400 vóxeles ED
    v = region_volumes(m, voxel_volume_mm3=2.0)
    assert v["v_ncr"] == 400 and v["v_et"] == 600 and v["v_ed"] == 800
    assert v["necrosis_fraction_core"] == pytest.approx(0.4)
    m[m == 4] = 3  # convención BraTS >= 2023
    assert region_volumes(m)["necrosis_fraction_core"] == pytest.approx(0.4)


def test_region_volumes_empty_and_bad_labels():
    assert np.isnan(region_volumes(np.zeros((3, 3, 3)))["necrosis_fraction_core"])
    with pytest.raises(ValueError):
        region_volumes(np.full((3, 3, 3), 7))


def test_imaging_table_reads_nifti(tmp_path):
    nib = pytest.importorskip("nibabel")
    from btc.radiogenomics.imaging import imaging_table

    for pid in ["TCGA-02-0006", "TCGA-08-0244"]:
        m = np.zeros((8, 8, 8), np.uint8)
        m[:2], m[2:4], m[4:6] = 1, 4, 2
        nib.save(nib.Nifti1Image(m, np.eye(4)), tmp_path / f"{pid}_2006.01.01_seg.nii.gz")
    df = imaging_table(tmp_path, "*_seg.nii.gz", "tcga")
    assert list(df.index) == ["TCGA-02-0006", "TCGA-08-0244"]
    assert (df["necrosis_fraction_core"] == 0.5).all()


# --- firmas ---------------------------------------------------------------------
def test_gmt_roundtrip(tmp_path):
    write_gmt({"A": ["g1", "G2", "g1"]}, tmp_path / "x.gmt")
    assert read_gmt(tmp_path / "x.gmt") == {"A": ["G1", "G2"]}


def test_scores_track_latent_signal():
    c = make_cohort(80, effect=0.0, seed=3)
    truth, genes = c["truth"]["hypoxia"], c["signatures"]["HYPOXIA_SYNTH"]
    for scorer in (score_mean_z, score_singscore):
        s = scorer(c["layers"]["rna"], genes)
        assert s.corr(truth, method="spearman") > 0.7


def test_score_requires_minimum_coverage():
    c = make_cohort(30, seed=4)
    with pytest.raises(ValueError, match="genes de la firma"):
        score_mean_z(c["layers"]["rna"], ["NOPE1", "NOPE2"])


# --- estadística ----------------------------------------------------------------
def test_partial_spearman_removes_confounder():
    rng = np.random.default_rng(0)
    conf = rng.normal(size=200)
    idx = pd.RangeIndex(200)
    x = pd.Series(conf + rng.normal(0, 0.5, 200), idx)
    y = pd.Series(conf + rng.normal(0, 0.5, 200), idx)
    raw = partial_spearman(x, y, None, n_perm=0, n_boot=0)
    adj = partial_spearman(x, y, pd.DataFrame({"c": conf}, idx), n_perm=500, n_boot=0)
    assert raw.rho > 0.6  # correlación espuria por el factor de confusión
    assert abs(adj.rho) < 0.15  # desaparece al ajustar
    assert adj.p_perm > 0.05


def test_permutation_pvalue_is_calibrated_under_null():
    """Bajo la hipótesis nula, ~5 % de p < 0.05 (no más)."""
    rng = np.random.default_rng(1)
    ps = []
    for _ in range(200):
        x, y = pd.Series(rng.normal(size=50)), pd.Series(rng.normal(size=50))
        ps.append(
            partial_spearman(x, y, None, n_perm=200, n_boot=0, seed=int(rng.integers(1e9))).p_perm
        )
    assert 0.01 <= np.mean(np.array(ps) < 0.05) <= 0.10


def test_bh_fdr_known_values():
    # ordenados: 0.01·4/1=0.04, 0.03·4/2=0.06, 0.04·4/3=0.0533, 0.20·4/4=0.20
    # mínimo acumulado desde el final -> 0.04, 0.0533, 0.0533, 0.20
    q = bh_fdr(np.array([0.01, 0.04, 0.03, 0.20]))
    np.testing.assert_allclose(q, [0.04, 0.16 / 3, 0.16 / 3, 0.20], rtol=1e-6)
    assert np.isnan(bh_fdr(np.array([np.nan, 0.5]))[0])


def test_layer_association_handles_missing_values():
    c = make_cohort(80, effect=0.8, seed=5)
    x = c["imaging"]["necrosis_fraction_core"]
    res = layer_association(x, c["layers"]["protein"], c["imaging"][["log_v_total"]])
    assert {"rho", "p", "q", "n"} <= set(res.columns)
    assert res["n"].min() < 80  # algunas proteínas con ausentes usan casos completos


def test_compare_and_pool_and_verdicts():
    same = compare_and_pool(0.35, 100, 0.30, 60, k=2)
    assert same["heterogeneity_p"] > 0.5 and 0.30 < same["pooled_rho"] < 0.35
    assert validation_verdict(0.30, 0.01, same["heterogeneity_p"]) == "replica"
    assert validation_verdict(0.20, 0.07, 0.5) == "consistente, potencia insuficiente"
    assert validation_verdict(-0.10, 0.80, 0.01) == "no replica"


# --- cargadores (sin red: se simula la API) --------------------------------------
def test_flatten_cptac_multiindex():
    cols = pd.MultiIndex.from_tuples(
        [
            ("VEGFA", "S1", "pep", "ENSP1"),
            ("VEGFA", "S2", "pep", "ENSP1"),
            ("CA9", "S5", "pep", "ENSP2"),
        ],
        names=["Name", "Site", "Peptide", "Database_ID"],
    )
    df = pd.DataFrame(
        [[1.0, 3.0, 5.0], [2.0, 4.0, 6.0]], index=["C3L-00016", "C3N-00002.N"], columns=cols
    )
    gene = flatten_cptac(df, "gene")
    assert list(gene.columns) == ["CA9", "VEGFA"] and gene.loc["C3L-00016", "VEGFA"] == 2.0
    site = flatten_cptac(df, "site")
    assert "VEGFA|S2" in site.columns and list(site.index) == ["C3L-00016", "C3N-00002"]


def test_cbioportal_client_with_fake_http():
    calls = []

    def fake(url, payload=None, timeout=0):
        calls.append(url)
        if "genes/fetch" in url:
            return [
                {"hugoGeneSymbol": "VEGFA", "entrezGeneId": 7422},
                {"hugoGeneSymbol": "CA9", "entrezGeneId": 768},
            ]
        if "molecular-data" in url:
            return [
                {"sampleId": "TCGA-02-0006-01", "entrezGeneId": 7422, "value": 9.1},
                {"sampleId": "TCGA-02-0006-01", "entrezGeneId": 768, "value": 7.3},
                {"sampleId": "TCGA-02-0006-10", "entrezGeneId": 768, "value": 1.0},
            ]
        if "sample-ids" in url:
            return ["TCGA-02-0006-01", "TCGA-08-0244-01"]
        if "mutations" in url:
            return [{"sampleId": "TCGA-08-0244-01"}]
        raise AssertionError(url)

    cb = CBioPortal(http=fake)
    expr = cb.expression(["VEGFA", "CA9", "FAKEGENE"])
    assert expr.shape == (1, 2) and expr.loc["TCGA-02-0006", "CA9"] == 7.3  # sin la muestra normal
    idh = cb.idh_status()
    assert idh.to_dict() == {"TCGA-02-0006": "wildtype", "TCGA-08-0244": "mutant"}


# --- estudio completo -------------------------------------------------------------
def _run(tmp_path, effect):
    from btc.radiogenomics.analysis import CohortData, StudyConfig, run_study

    d = make_cohort(100, effect=effect, seed=21)
    v = make_cohort(70, effect=effect, seed=22, prefix="VAL")
    cfg = StudyConfig(
        primary_signature="HYPOXIA_SYNTH",
        sensitivity_signatures=["HYPOXIA_ALT"],
        n_perm=1000,
        n_boot=200,
        n_random_sets=100,
    )
    return run_study(
        CohortData("D", d["imaging"], d["clinical"], {"rna": d["layers"]["rna"]}),
        d["signatures"],
        cfg,
        tmp_path,
        validation=CohortData(
            "V", v["imaging"], v["clinical"].join(v["layers"]["purity"]), v["layers"]
        ),
    )


def test_study_recovers_planted_effect(tmp_path):
    rep = _run(tmp_path, effect=0.9)
    assert rep["discovery"]["confirmatory_success"]
    assert rep["validation"]["layers"]["rna"]["verdict"] == "replica"
    assert rep["validation"]["layers"]["protein"]["test"]["coverage"]["fraction"] < 1
    assert "with_purity" in rep["validation"]["layers"]["rna"]
    for f in [
        "informe.json",
        "resumen.md",
        "confirmatorio_residuos.png",
        "control_firmas_aleatorias.png",
        "exploratorio_V_protein.csv",
    ]:
        assert (tmp_path / f).exists(), f


def test_study_does_not_invent_effect(tmp_path):
    """Sin efecto plantado el estudio no debe declarar éxito (seed fijo: determinista)."""
    rep = _run(tmp_path, effect=0.0)
    assert not rep["discovery"]["confirmatory_success"]


def test_false_positive_rate_is_nominal():
    """Calibración del test confirmatorio sobre muchas cohortes nulas.

    Con una sola cohorte nula, un 5 % de las veces saldrá "significativo" por
    azar: exigir que una semilla concreta no lo haga sería un test frágil. Lo
    correcto es comprobar que, sobre muchas cohortes, la tasa de falsos positivos
    ronda el 5 % y la ρ media es ~0 (método insesgado).
    """
    from btc.radiogenomics.signatures import score_mean_z

    rhos, ps = [], []
    for s in range(60):
        c = make_cohort(80, effect=0.0, seed=1000 + s, n_genes=300)
        score = score_mean_z(c["layers"]["rna"], c["signatures"]["HYPOXIA_SYNTH"])
        t = c["imaging"].join(c["clinical"])
        r = partial_spearman(
            t["necrosis_fraction_core"],
            score,
            t[["age", "log_v_total"]],
            n_perm=300,
            n_boot=0,
            seed=s,
        )
        rhos.append(r.rho)
        ps.append(r.p_perm)
    assert abs(np.mean(rhos)) < 0.05
    assert np.mean(np.array(ps) < 0.05) <= 0.15  # 5 % nominal + margen por n=60 cohortes
