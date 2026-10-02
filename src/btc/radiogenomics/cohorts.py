"""Identificadores de paciente, solapamiento imagen ↔ ómica, potencia y descargas.

Todo el cruce entre fuentes depende de que el mismo paciente tenga el mismo ID
en todas ellas. Esto parece trivial y es la fuente de errores más habitual:

- TCGA usa barcodes jerárquicos: ``TCGA-02-0006-01A-01R-...``. Los 12 primeros
  caracteres (``TCGA-02-0006``) identifican al PACIENTE; lo que sigue identifica
  la muestra (``-01`` = tumor primario, ``-10`` = sangre normal...), la porción y
  el analito. Si no se recorta, el join imagen ↔ ómica da cero coincidencias.
- CPTAC usa ``C3L-00016`` / ``C3N-01234``; el paquete ``cptac`` puede añadir
  sufijos (``.N`` para tejido normal).
"""

from __future__ import annotations

import json
import math
import re
import urllib.request
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from btc.utils import get_logger

log = get_logger(__name__)

_TCGA_RE = re.compile(r"^(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})", re.IGNORECASE)
_CPTAC_RE = re.compile(r"(C3[LN]-\d{5})", re.IGNORECASE)


# ----------------------------------------------------------------------------
# Identificadores
# ----------------------------------------------------------------------------
def normalize_patient_id(raw: str, cohort: str) -> str:
    """Convierte cualquier ID de muestra/fichero al ID de paciente canónico.

    >>> normalize_patient_id("TCGA-02-0006-01A-01R", "tcga")
    'TCGA-02-0006'
    >>> normalize_patient_id("c3l-00016.N", "cptac")
    'C3L-00016'
    """
    raw = str(raw).strip()
    regex = {"tcga": _TCGA_RE, "cptac": _CPTAC_RE}.get(cohort.lower())
    if regex is None:
        return raw
    m = regex.search(raw)
    if not m:
        raise ValueError(f"'{raw}' no parece un ID de {cohort.upper()}")
    return m.group(1).upper()


def is_primary_tumor_sample(sample_id: str) -> bool:
    """En TCGA, el código de tipo de muestra ``01`` es tumor sólido primario."""
    parts = str(sample_id).split("-")
    return len(parts) >= 4 and parts[3][:2] == "01"


# ----------------------------------------------------------------------------
# Factibilidad: solapamiento y potencia
# ----------------------------------------------------------------------------
def min_detectable_r(n: int, alpha: float = 0.05, power: float = 0.80) -> float:
    """Correlación mínima detectable con ``n`` pacientes (test bilateral).

    Se basa en la transformación z de Fisher: z = atanh(r) es aproximadamente
    normal con varianza 1/(n-3). Para detectar r con potencia 1-β hace falta
    atanh(r)·sqrt(n-3) ≥ z_{1-α/2} + z_{1-β}. Despejando r:

        r_min = tanh( (z_{1-α/2} + z_{1-β}) / sqrt(n - 3) )

    Con covariables la n efectiva baja en el nº de covariables, por eso el
    valor es ligeramente optimista; con 2 covariables la diferencia es mínima.
    """
    if n <= 3:
        return float("nan")
    z = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)
    return float(math.tanh(z / math.sqrt(n - 3)))


def n_for_r(r: float, alpha: float = 0.05, power: float = 0.80) -> int:
    """Pacientes necesarios para detectar una correlación ``r`` (inversa de la anterior)."""
    z = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)
    return math.ceil((z / math.atanh(r)) ** 2 + 3)


@dataclass
class OverlapReport:
    n_imaging: int
    n_omics: int
    n_overlap: int
    min_detectable_r: float
    decision: str
    only_imaging: list[str]

    def __str__(self) -> str:
        return (
            f"Pacientes con imagen: {self.n_imaging}\n"
            f"Pacientes con ómica:  {self.n_omics}\n"
            f"En ambos:             {self.n_overlap}\n"
            f"Correlación mínima detectable (α=0.05, potencia 80 %): {self.min_detectable_r:.3f}\n"
            f"Decisión según el plan: {self.decision}"
        )


def overlap_report(imaging_ids: Iterable[str], omics_ids: Iterable[str]) -> OverlapReport:
    """Cruza dos listas de IDs YA normalizados y aplica la regla de decisión del plan."""
    img, omx = set(imaging_ids), set(omics_ids)
    both = img & omx
    n = len(both)
    if n >= 85:
        decision = "PROCEDER"
    elif n >= 60:
        decision = "PROCEDER declarando que solo se detectan efectos moderados"
    else:
        decision = "DETENER y replantear (n < 60)"
    return OverlapReport(len(img), len(omx), n, min_detectable_r(n), decision, sorted(img - omx))


# ----------------------------------------------------------------------------
# cBioPortal (TCGA): API REST pública, sin registro
# ----------------------------------------------------------------------------
CBIO_BASE = "https://www.cbioportal.org/api"


def _http_json(url: str, payload: object | None = None, timeout: int = 120) -> object:
    """GET (sin payload) o POST JSON. Aislado en una función para poder simularlo en tests."""
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json", "Accept": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - URL fija
        return json.loads(resp.read())


class CBioPortal:
    """Cliente mínimo de la API de cBioPortal para un estudio (por defecto ``gbm_tcga``).

    Endpoints usados (documentación: https://www.cbioportal.org/api/swagger-ui):
    - ``GET  /sample-lists/{id}/sample-ids``
    - ``POST /genes/fetch``                              símbolo HUGO → Entrez
    - ``POST /molecular-profiles/{id}/molecular-data/fetch``
    - ``GET  /studies/{id}/clinical-data``
    - ``POST /molecular-profiles/{id}/mutations/fetch``
    """

    def __init__(self, study: str = "gbm_tcga", base: str = CBIO_BASE, http=_http_json):
        self.study, self.base, self._http = study, base, http

    def sample_ids(self, sample_list: str) -> list[str]:
        return list(self._http(f"{self.base}/sample-lists/{sample_list}/sample-ids"))

    def entrez_ids(self, symbols: list[str]) -> dict[str, int]:
        genes = self._http(f"{self.base}/genes/fetch?geneIdType=HUGO_GENE_SYMBOL", symbols)
        return {g["hugoGeneSymbol"]: int(g["entrezGeneId"]) for g in genes}

    def expression(
        self,
        symbols: list[str],
        profile: str = "gbm_tcga_mrna_U133",
        sample_list: str = "gbm_tcga_mrna_U133",
        chunk: int = 500,
    ) -> pd.DataFrame:
        """Matriz pacientes × genes del perfil indicado (solo tumores primarios).

        Se pide por lotes de ``chunk`` genes para no superar el tamaño de petición.
        Los genes que cBioPortal no reconoce se registran en el log (no se pierden
        en silencio): la cobertura de la firma se reporta después.
        """
        mapping = self.entrez_ids(symbols)
        missing = sorted(set(symbols) - set(mapping))
        if missing:
            log.warning(
                "%d símbolos sin Entrez en cBioPortal (p. ej. %s)", len(missing), missing[:5]
            )
        inv = {v: k for k, v in mapping.items()}
        rows: list[dict] = []
        ids = list(mapping.values())
        for i in range(0, len(ids), chunk):
            rows += self._http(
                f"{self.base}/molecular-profiles/{profile}/molecular-data/fetch?projection=SUMMARY",
                {"sampleListId": sample_list, "entrezGeneIds": ids[i : i + chunk]},
            )
        df = pd.DataFrame(rows)
        if df.empty:
            raise ValueError("cBioPortal no devolvió datos de expresión")
        df = df[df["sampleId"].map(is_primary_tumor_sample)]
        df["gene"] = df["entrezGeneId"].map(inv)
        df["patient_id"] = df["sampleId"].map(lambda s: normalize_patient_id(s, "tcga"))
        return df.pivot_table(index="patient_id", columns="gene", values="value", aggfunc="mean")

    def clinical(self) -> pd.DataFrame:
        """Tabla ancha pacientes × atributos clínicos (edad, supervivencia...)."""
        rows = self._http(
            f"{self.base}/studies/{self.study}/clinical-data?clinicalDataType=PATIENT&projection=SUMMARY"
        )
        df = pd.DataFrame(rows).pivot_table(
            index="patientId", columns="clinicalAttributeId", values="value", aggfunc="first"
        )
        df.index = [normalize_patient_id(p, "tcga") for p in df.index]
        df.index.name = "patient_id"
        return df

    def idh_status(self) -> pd.Series:
        """``mutant`` / ``wildtype`` / ausente (= desconocido) por paciente.

        Wildtype solo para pacientes SECUENCIADOS sin mutación en IDH1/IDH2. Un
        paciente no secuenciado no es wildtype: es desconocido. Confundir ambas
        cosas metería IDH-mutantes no detectados en el grupo "GBM".
        """
        sequenced = {
            normalize_patient_id(s, "tcga") for s in self.sample_ids(f"{self.study}_sequenced")
        }
        muts = self._http(
            f"{self.base}/molecular-profiles/{self.study}_mutations/mutations/fetch?projection=SUMMARY",
            {
                "sampleListId": f"{self.study}_sequenced",
                "entrezGeneIds": [3417, 3418],
            },  # IDH1, IDH2
        )
        mutant = {normalize_patient_id(m["sampleId"], "tcga") for m in muts}
        status = pd.Series("wildtype", index=sorted(sequenced), name="idh_status")
        status[status.index.isin(mutant)] = "mutant"
        status.index.name = "patient_id"
        return status


# ----------------------------------------------------------------------------
# CPTAC: paquete ``cptac`` (descarga y cachea los datos la primera vez)
# ----------------------------------------------------------------------------
# Capa -> (fuente por defecto, nivel). "gene" agrega por símbolo; "site" conserva
# el sitio de modificación (fosforilación/acetilación) en el nombre.
CPTAC_LAYERS: dict[str, tuple[str, str]] = {
    "transcriptomics": ("bcm", "gene"),  # RNA-seq, log2(UQ+1)
    "proteomics": ("umich", "gene"),  # abundancia proteica normalizada
    "phosphoproteomics": ("umich", "site"),
    "acetylproteomics": ("umich", "site"),
    "CNV": ("washu", "gene"),
    "miRNA": ("washu", "gene"),
    "xcell": ("washu", "gene"),  # deconvolución de tipos celulares
    "cibersort": ("washu", "gene"),
    "tumor_purity": ("washu", "gene"),
}


def flatten_cptac(df: pd.DataFrame, level: str = "gene") -> pd.DataFrame:
    """Aplana las columnas MultiIndex de ``cptac`` a nombres simples.

    ``cptac`` devuelve columnas como (Name, Database_ID) o, en fosfoproteómica,
    (Name, Site, Peptide, Database_ID). Varias isoformas/péptidos pueden
    compartir nombre: se promedian para tener UNA columna por gen (o por sitio).
    """
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        names = list(df.columns.names)
        gene = (
            df.columns.get_level_values("Name")
            if "Name" in names
            else df.columns.get_level_values(0)
        )
        if level == "site" and "Site" in names:
            new = [
                f"{g}|{s}" for g, s in zip(gene, df.columns.get_level_values("Site"), strict=True)
            ]
        else:
            new = list(gene)
        df.columns = new
    df = df.apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all")
    df = df.T.groupby(level=0).mean().T  # promedia columnas con el mismo nombre
    df.index = [normalize_patient_id(i, "cptac") for i in df.index]
    df.index.name = "patient_id"
    return df.groupby(level=0).mean()  # por si un paciente aparece dos veces


def load_cptac_layer(layer: str, source: str | None = None) -> pd.DataFrame:
    """Descarga (la 1ª vez) y devuelve una capa ómica de CPTAC-GBM, solo tumores.

    Requiere ``pip install cptac`` y conexión a internet la primera vez.
    """
    import cptac  # dependencia opcional

    default_source, level = CPTAC_LAYERS[layer]
    gbm = cptac.Gbm()
    df = gbm.get_dataframe(layer, source or default_source, tissue_type="tumor")
    return flatten_cptac(df, level)


def load_feature_table(path: str, cohort: str, id_col: str = "patient_id") -> pd.DataFrame:
    """Cargador genérico (CSV/TSV) para capas que no están en ``cptac``.

    Por ejemplo el metaboloma o el lipidoma de CPTAC-GBM, que se descargan de
    las tablas suplementarias. Formato esperado: una fila por paciente, una
    columna ``id_col`` y el resto numéricas.
    """
    sep = "\t" if path.endswith((".tsv", ".txt", ".tsv.gz")) else ","
    df = pd.read_csv(path, sep=sep)
    df.index = [normalize_patient_id(i, cohort) for i in df.pop(id_col)]
    df.index.name = "patient_id"
    return df.apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all")


def ensure_log_scale(expr: pd.DataFrame) -> pd.DataFrame:
    """Aplica log2(x+1) si la matriz parece estar en escala lineal.

    Heurística: los valores de expresión en log2 rara vez superan ~25. Si el
    percentil 99 es mayor que 100, casi seguro está en escala lineal (FPKM,
    TPM, intensidades crudas). Se registra la decisión para que sea auditable.
    """
    q99 = float(np.nanpercentile(expr.to_numpy(dtype=float), 99))
    if q99 > 100:
        log.info("Percentil 99 = %.1f: se aplica log2(x+1)", q99)
        return np.log2(expr.clip(lower=0) + 1)
    return expr
