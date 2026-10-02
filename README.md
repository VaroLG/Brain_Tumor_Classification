# Clasificación de tumores cerebrales en RM con transfer learning

[![CI](https://github.com/VaroLG/Brain_Tumor_Classification/actions/workflows/ci.yml/badge.svg)](https://github.com/VaroLG/Brain_Tumor_Classification/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

> **EN —** Brain-tumour MRI classification with ImageNet transfer learning (EfficientNet, VGG19, …):
> tumour subtype, glioma grade, and glioblastoma overall survival. Started as my MSc thesis
> (Bioinformatics & Biostatistics, UOC, 2025) and refactored into a reproducible package with
> **patient-level splits**, train-only augmentation, two-phase fine-tuning, and patient-level
> evaluation with bootstrap confidence intervals.

Código del Trabajo Fin de Máster *«Clasificación de tumores cerebrales mediante redes neuronales
convolucionales con transferencia de aprendizaje»* (Máster en Bioinformática y Bioestadística,
UOC, enero de 2025), reescrito como paquete Python reproducible. La memoria completa está en
[`docs/memoria_tfm.pdf`](docs/memoria_tfm.pdf).

## Tareas

| Tarea | Dataset | Clases | Config |
|---|---|---|---|
| Subtipo tumoral | [Kaggle Brain Tumor Classification](https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri) | glioma · meningioma · hipofisario · sin tumor | `configs/subtypes_kaggle.yaml` |
| Grado de glioma | [REMBRANDT (TCIA)](https://www.cancerimagingarchive.net/collection/rembrandt/) | II · III · IV | `configs/glioma_grade_rembrandt.yaml` |
| Bajo vs alto grado | REMBRANDT | LGG · HGG | `configs/glioma_lgg_hgg_rembrandt.yaml` |
| Supervivencia global en GBM | [UPENN-GBM (TCIA)](https://www.cancerimagingarchive.net/collection/upenn-gbm/) | corta · larga (umbral = mediana, con censura) | `configs/gbm_survival_upenn.yaml` |
| IDH1 / MGMT en GBM *(nuevo)* | UPENN-GBM | wildtype · mutant / no metilado · metilado | `configs/gbm_idh_upenn.yaml`, `configs/gbm_mgmt_upenn.yaml` |

## Estudio radiogenómico: necrosis en RM e hipoxia multiómica *(nuevo)*

Extensión del TFM hacia las ómicas con una pregunta concreta: **¿la proporción de tumor
necrótico que se ve en la RM refleja el programa molecular de hipoxia, a nivel de ARN y de
proteína?** El plan está prerregistrado (hipótesis, variables, tests y criterios de éxito fijados
antes de ver los datos) en [`docs/PLAN_ANALISIS_RADIOGENOMICA.md`](docs/PLAN_ANALISIS_RADIOGENOMICA.md).

| Fase | Cohorte | Datos | Qué se afirma |
|---|---|---|---|
| Confirmatorio | TCGA-GBM | Segmentación experta BraTS + microarray U133A | Un único test: ρ parcial de Spearman necrosis ↔ `HALLMARK_HYPOXIA`, ajustado por edad y volumen |
| Robustez | TCGA-GBM | Ídem | Otro método de puntuación, otra firma, control con 1 000 firmas aleatorias |
| Validación | CPTAC-GBM | RNA-seq **y proteoma** | Replica / consistente pero sin potencia / no replica + estimación combinada |
| Exploratorio | Ambas | Todas las capas: transcriptoma, proteoma, fosfo- y acetiloma, CNV, miRNA, tipos celulares | Hipótesis con FDR, etiquetadas como tales |

```bash
pip install -e ".[omics]"
python scripts/demo_radiogenomics.py           # todo el estudio sobre cohortes sintéticas (~20 s)

# Con datos reales (en tu ordenador):
btc rg-imaging --seg-root data/BraTS-TCGA-GBM --out data/rg/tcga_imaging.csv
btc rg-overlap --imaging data/rg/tcga_imaging.csv --cbioportal-list gbm_tcga_mrna_U133   # ¡primero!
btc rg-fetch-tcga --gmt data/msigdb/h.all.Hs.symbols.gmt data/msigdb/c2.cgp.Hs.symbols.gmt --out-dir data/rg
btc rg-fetch-cptac --out-dir data/rg
btc rg-analyze --config configs/radiogenomics_necrosis_hypoxia.yaml --out-dir runs/radiogenomica
```

`rg-overlap` aplica la regla de factibilidad del plan (n ≥ 85 procede; n < 60 se detiene) antes
de descargar nada más. El código se valida sobre cohortes sintéticas con efecto conocido: recupera
el efecto cuando existe y su tasa de falsos positivos sin efecto es la nominal (≈ 5 %).

## Qué cambia respecto al código original del TFM

Los notebooks originales se conservan en [`legacy/tfm_notebooks/`](legacy/tfm_notebooks/) para
trazabilidad. Al revisarlos para publicarlos encontré varios problemas metodológicos que afectan a
las métricas; el detalle de cada uno y su corrección está en
[`docs/METODOLOGIA.md`](docs/METODOLOGIA.md). Resumen:

| Problema en el TFM | Efecto | Corrección |
|---|---|---|
| Split train/val/test por **imagen** | Cortes casi idénticos del mismo paciente en train y test → métricas optimistas | Split **por paciente** (`StratifiedGroupKFold`) + comprobación automática de fuga |
| Aumento de datos también en val/test | Evaluación aleatoria, no reproducible | Aumento solo en train |
| VGG19 con `rescale=1/255` | Entrada fuera del rango para el que se preentrenó | `preprocess_input` específico de cada backbone |
| SG binarizada por la media sin censura | Pacientes vivos con seguimiento corto etiquetados como "SG baja" | Umbral = mediana; censurados antes del umbral se excluyen |
| Selección manual de cortes con tumor | Lento, subjetivo, no reproducible | Selección automática con la máscara de segmentación de UPENN |
| Cabeza Dense 2560→1280→1024 | ~6 M parámetros nuevos para pocos cientos de pacientes → sobreajuste | GAP → Dropout → Dense(256) → Dropout |
| Nº de épocas fijado a mano | 60 épocas peor que 15 en validación | EarlyStopping + fine-tuning en 2 fases |
| Métricas por imagen, sin incertidumbre | No se puede comparar modelos | Métricas por paciente + IC 95 % por bootstrap |
| Sin semillas, rutas de Windows, código triplicado | No reproducible | Config YAML, semillas, CLI, tests y CI |

### Resultados del TFM (histórico)

Valores publicados en la memoria, **con split por imagen**: es probable que estén inflados por la
fuga de datos descrita arriba y no deben compararse con los que produzca esta versión.

| Tarea | Mejor modelo | Resultado en test |
|---|---|---|
| Subtipo (4 clases) | EfficientNetB1 | F1: glioma 0.53 · meningioma 0.70 · sin tumor 0.78 · hipofisario 0.77 |
| Grado II/III/IV | VGG19 | F1: II 0.69 · III 0.65 · IV 0.79; AUC ≈ 0.83 |
| LGG vs HGG | EfficientNetB1 | F1: HGG 0.86 · LGG 0.68; AUC 0.84 |
| SG alta/baja en GBM | VGG19 | F1: alta 0.46 · baja 0.75; AUC 0.63 |

Los resultados con la metodología corregida se añadirán a esta sección al reentrenar con los
datos originales.

## Instalación

```bash
git clone https://github.com/VaroLG/Brain_Tumor_Classification.git
cd Brain_Tumor_Classification
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[medical]"        # añade ".[dev]" para tests y linter
```

Prueba rápida sin descargar datos (dataset sintético, ~1–2 min en CPU):

```bash
python scripts/demo_synthetic.py
```

## Uso

Todo el flujo gira alrededor de un **manifest**: un CSV con una fila por corte 2D y las columnas
`image_path`, `patient_id`, `label` (y `split` tras el paso 2).

```text
datos originales ──► 1. btc manifest-*  ──► 2. btc split  ──► 3. btc train  ──► runs/<experimento>/
 (NIfTI/DICOM/PNG)     PNG + manifest.csv     por paciente      2 fases + test     métricas, figuras, modelo
```

### 1. Construir el manifest

```bash
# UPENN-GBM: supervivencia, IDH1 o MGMT (cortes elegidos con la máscara de segmentación)
btc manifest-upenn --nifti-root data/UPENN-GBM --clinical data/UPENN-GBM_clinical_info_v2.1.csv \
    --task survival --out-dir data/png/gbm_survival --manifest data/gbm_survival.csv
```

> Los nombres de columna del CSV clínico cambian entre versiones del dataset. Si el comando no las
> encuentra, muestra las disponibles; indícalas con `--id-col`, `--days-col`, `--event-col`
> (1 = fallecido, 0 = censurado) o `--value-col`.

```bash
# REMBRANDT: necesita un CSV patient_id,label con el grado de cada paciente
btc manifest-dicom --dicom-root data/REMBRANDT --labels data/rembrandt_grades.csv \
    --out-dir data/png/glioma_grade --manifest data/glioma_grade.csv

# Kaggle: carpetas por clase (se agrupan casi-duplicados con hash perceptual)
btc manifest-folders --root data/kaggle_brain_tumor --manifest data/subtypes_kaggle.csv
```

### 2. Split por paciente

```bash
btc split --manifest data/gbm_survival.csv --val 0.15 --test 0.20 --seed 42
```

### 3. Entrenar y evaluar

```bash
btc train --config configs/gbm_survival_upenn.yaml
btc evaluate --run-dir runs/gbm_survival_upenn_<fecha>   # re-evaluar un run guardado
```

Cada run guarda `config.yaml`, el `manifest.csv` usado, `model.keras`, el historial de cada fase,
`metrics_test.json` (por corte y por paciente, con IC 95 %), `predictions_patient.csv` y las
figuras de matriz de confusión y ROC.

## Estructura

```text
src/btc/
├── data/
│   ├── nifti.py          # UPENN-GBM: cortes con tumor según la máscara
│   ├── dicom.py          # REMBRANDT: series DICOM → PNG con PatientID
│   ├── folders.py        # Kaggle: carpetas por clase + agrupación por dHash
│   ├── labels.py         # supervivencia con censura, IDH1, MGMT
│   └── preprocessing.py  # normalización por percentiles, recorte de fondo
├── splits.py             # split por paciente y detección de fuga
├── datasets.py           # tf.data; aumento solo en train
├── models.py             # backbones + preprocess_input correcto + cabeza
├── train.py              # entrenamiento en 2 fases
├── evaluate.py           # métricas por paciente + bootstrap
├── synthetic.py          # datos sintéticos para tests y demo
├── radiogenomics/        # estudio necrosis ↔ hipoxia: imagen, cohortes, firmas, estadística
└── cli.py                # comando `btc`
configs/                  # un YAML por experimento
tests/                    # pytest (corre en CPU con datos sintéticos)
legacy/tfm_notebooks/     # notebooks originales del TFM
docs/                     # memoria, metodología y hoja de ruta
```

## Limitaciones

- **Kaggle no tiene identificador de paciente.** El agrupamiento por hash perceptual elimina
  duplicados, pero dos cortes distintos de un mismo paciente pueden seguir en splits distintos.
  Las métricas en esta tarea son optimistas.
- **REMBRANDT no tiene segmentaciones**: los cortes se eligen por la franja central del volumen o
  con un CSV de selección manual (`--selected-slices`).
- **Modelos 2D**: se pierde la información volumétrica. Ver la hoja de ruta.
- Proyecto de investigación y aprendizaje: **no es un dispositivo médico** ni debe usarse para
  decisiones clínicas.

## Hoja de ruta

Ver [`docs/ROADMAP_RADIOGENOMICA.md`](docs/ROADMAP_RADIOGENOMICA.md): reentrenar con la metodología
corregida, supervivencia como problema de supervivencia (Cox, C-index), modelos 3D y la integración
con datos ómicos (radiogenómica con TCGA/TCIA e Ivy GAP).

## Cita

```bibtex
@mastersthesis{linacero2025brain,
  author = {Linacero Gracia, Álvaro},
  title  = {Clasificación de tumores cerebrales mediante redes neuronales convolucionales
            con transferencia de aprendizaje},
  school = {Universitat Oberta de Catalunya},
  year   = {2025},
  note   = {Tutora: Karmele López de Ipiña}
}
```

Si usas los datasets, cita también sus publicaciones originales (UPENN-GBM: Bakas et al.,
*Sci Data* 2022; REMBRANDT y TCIA según sus políticas de uso).

## Licencia

Código bajo licencia [MIT](LICENSE). Los datos de imagen médica no se incluyen y tienen sus
propias licencias.
