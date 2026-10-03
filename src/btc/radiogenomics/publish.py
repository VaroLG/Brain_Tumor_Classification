"""Salvaguardas para publicar resultados sin exponer datos con licencia restringida.

Contexto (ver ``docs/DATOS_Y_LICENCIAS.md``): con la *TCIA Restricted License*,
las obras derivadas de las imágenes (máscaras, tablas de características por
paciente) heredan las restricciones de las imágenes. Los resultados AGREGADOS
(ρ, IC, figuras resumen) sí se pueden publicar.

Este módulo hace dos cosas sencillas pero que evitan el error más probable,
subir por descuido un CSV por paciente a GitHub:

1. Decide la carpeta de salida de las tablas por paciente según la licencia de
   la cohorte. Las restringidas van a ``solo_local/``, ignorada por git.
2. Escanea una carpeta buscando identificadores de pacientes de cohortes
   restringidas fuera de ``solo_local/``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

LOCAL_ONLY_DIR = "solo_local"

# Formatos de identificador de paciente por cohorte
ID_PATTERNS: dict[str, str] = {
    "cptac": r"C3[LN]-\d{5}",
    "tcga": r"TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}",
    "upenn": r"UPENN-GBM-\d{5}",
}

TEXT_EXTENSIONS = {".csv", ".tsv", ".txt", ".json", ".md", ".yaml", ".yml", ".html", ".svg"}

_README = """# Solo local — NO publicar

Esta carpeta contiene tablas por paciente derivadas de cohortes con licencia
restringida (p. ej. imágenes de CPTAC-GBM bajo la TCIA Restricted License).
Las obras derivadas heredan las restricciones de las imágenes: no se suben al
repositorio ni se comparten fuera de los colaboradores del acuerdo.
La carpeta está en .gitignore. Ver docs/DATOS_Y_LICENCIAS.md.
"""


def patient_output_dir(out_dir: str | Path, restricted: bool) -> Path:
    """Carpeta donde escribir salidas POR PACIENTE de una cohorte."""
    out_dir = Path(out_dir)
    if not restricted:
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir
    local = out_dir / LOCAL_ONLY_DIR
    local.mkdir(parents=True, exist_ok=True)
    (local / "LEEME.md").write_text(_README, encoding="utf-8")
    return local


@dataclass
class Finding:
    path: str
    n_ids: int
    example: str


def scan_for_patient_ids(root: str | Path, cohorts: list[str]) -> list[Finding]:
    """Busca IDs de las cohortes indicadas en ficheros de texto fuera de ``solo_local/``.

    Devuelve un hallazgo por fichero (lista vacía = nada que objetar). Las
    imágenes (PNG) no se inspeccionan: una figura de dispersión no contiene IDs.
    """
    patterns = [ID_PATTERNS[c] for c in cohorts if c in ID_PATTERNS]
    if not patterns:
        return []
    regex = re.compile("|".join(f"(?:{p})" for p in patterns))
    findings = []
    for path in sorted(Path(root).rglob("*")):
        if not path.is_file() or LOCAL_ONLY_DIR in path.parts:
            continue
        if path.suffix.lower() not in TEXT_EXTENSIONS:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        ids = set(regex.findall(text))
        if ids:
            findings.append(Finding(str(path), len(ids), sorted(ids)[0]))
    return findings
