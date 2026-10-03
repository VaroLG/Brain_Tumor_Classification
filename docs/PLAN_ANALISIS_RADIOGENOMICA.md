# Plan de análisis prerregistrado — Necrosis en RM e hipoxia multiómica en glioblastoma

> **Estado:** plan fijado *antes* de descargar o mirar los datos reales.
> Cualquier cambio posterior se registra en la sección final («Desviaciones»)
> con fecha y motivo. Este documento es lo que separa un análisis confirmatorio
> de una búsqueda de resultados a posteriori.

## 1. Pregunta

¿La proporción de tumor necrótico que se ve en la resonancia magnética refleja
la activación del programa molecular de hipoxia del glioblastoma, y se
mantiene esa relación a nivel de ARN **y de proteína**?

**Racional biológico.** El glioblastoma prolifera más rápido que su
vascularización. Las regiones alejadas de los vasos se quedan sin oxígeno, se
activa HIF-1α y su programa transcripcional (VEGFA, CA9, SLC2A1, genes
glucolíticos…), y si la hipoxia es extrema las células mueren: aparece la
necrosis, visible en T1 con contraste como un núcleo sin realce rodeado de un
anillo que sí realza. Si la imagen captura esta biología, los tumores con más
necrosis deberían tener mayor puntuación de hipoxia.

**Trabajo previo.** La asociación necrosis–hipoxia en TCGA se ha descrito antes
con microarrays (p. ej. Colen et al.). La aportación de este estudio no es la
hipótesis, sino: (1) un análisis confirmatorio con criterios fijados de
antemano, (2) validación externa en una cohorte independiente, (3) validación a
nivel de **proteína** (CPTAC), que no existía en los trabajos originales, y (4)
la comprobación de si una segmentación automática propia reproduce el
resultado obtenido con segmentaciones de expertos.

## 2. Cohortes

| Rol | Cohorte | Imagen | Ómica | Enlace |
|---|---|---|---|---|
| Descubrimiento | TCGA-GBM | BraTS-TCGA-GBM: RM preoperatoria con segmentación revisada por neurorradiólogo (102 pacientes disponibles) | Microarray Affymetrix U133A (528 muestras en cBioPortal `gbm_tcga`), clínica, mutaciones | Barcode TCGA (12 caracteres) |
| Validación | CPTAC-GBM | RM en TCIA (66 pacientes); segmentación de BraTS 2021 cuando exista, o automática | RNA-seq, proteoma, fosfoproteoma, acetiloma, CNV, mutaciones, miRNA, pureza tumoral, deconvolución inmune (paquete `cptac`) | ID CPTAC (`C3L-…`/`C3N-…`) |

**Por qué microarray y no RNA-seq en TCGA.** Solo 166 muestras de TCGA-GBM
tienen RNA-seq frente a 528 con U133A, y los pacientes con RM en TCIA proceden
mayoritariamente de la fase inicial del proyecto, perfilada con microarray.
Las puntuaciones de firmas por muestra son válidas en ambas plataformas.

### Criterios de inclusión / exclusión
- RM preoperatoria con segmentación disponible.
- Expresión disponible en la plataforma de la cohorte.
- **Excluir IDH-mutantes conocidos** (OMS 2021: ya no son «glioblastoma»).
  Los pacientes con IDH desconocido se incluyen en el análisis principal y se
  excluyen en un análisis de sensibilidad.
- Muestras de tumor primario (código de muestra TCGA `-01`).

### Criterio de factibilidad (se evalúa antes de cualquier análisis)
Con un test bilateral, α = 0,05 y potencia del 80 %:

| n en el solapamiento | Correlación mínima detectable |
|---|---|
| 100 | 0,28 |
| 85 | 0,30 |
| 60 | 0,35 |

- n ≥ 85 → se procede.
- 60 ≤ n < 85 → se procede declarando que solo se detectan efectos moderados.
- n < 60 → **se detiene** y se replantea.

Se calcula con `btc rg-overlap` y queda registrado en la sección de desviaciones.

## 3. Variables

### Exposición (imagen)
A partir de la segmentación con etiquetas BraTS:
`1` = núcleo necrótico / no realzante (NCR), `2` = edema (ED),
`4` (o `3` en BraTS ≥ 2023) = tumor que realza (ET).

- **Primaria:** fracción necrótica del núcleo tumoral = V_NCR / (V_NCR + V_ET).
  Se usa el núcleo (y no el tumor completo) porque el edema varía mucho entre
  pacientes por motivos ajenos a la hipoxia del tumor.
- Secundaria: log(V_NCR + 1) en mm³.

*Limitación declarada:* en BraTS la etiqueta 1 agrupa necrosis y tumor no
realzante; la variable es un proxy de necrosis, no necrosis histológica.

### Resultado (ómica)
- **Primario:** puntuación por muestra de la firma `HALLMARK_HYPOXIA` (MSigDB),
  calculada como la media de los z-scores de sus genes (cada gen estandarizado
  en la cohorte). Elegida por ser simple, transparente y no depender de la
  composición del resto de genes.
- Sensibilidad: método por rangos (singscore) y una segunda firma de hipoxia
  independiente (`BUFFA_HYPOXIA_METAGENE`).

### Covariables (fijadas de antemano)
- Edad al diagnóstico.
- log del volumen tumoral total (V_NCR + V_ED + V_ET): evita que «más necrosis»
  signifique simplemente «tumor más grande».
- En CPTAC, análisis de sensibilidad añadiendo **pureza tumoral** (ESTIMATE):
  un tumor muy necrótico puede tener menos células tumorales en la muestra, lo
  que diluiría cualquier señal.

## 4. Análisis

### 4.1 Confirmatorio (UN único test)
Correlación **parcial de Spearman** entre la fracción necrótica y la puntuación
de hipoxia, ajustando por las covariables:

1. Transformar todas las variables a rangos.
2. Regresar (MCO) los rangos de exposición y de resultado sobre los rangos de
   las covariables.
3. Correlación de Pearson entre los dos residuos = ρ parcial.

- Valor p por **permutación** de residuos (10 000 permutaciones, bilateral).
- IC 95 % por **bootstrap** de pacientes (5 000 remuestreos, percentil).
- **Éxito confirmatorio:** ρ > 0 y p < 0,05 en TCGA.

Al haber un solo test principal no se corrige por comparaciones múltiples.

### 4.2 Robustez (descriptivos, no cambian la conclusión principal)
- **Control negativo:** 1 000 firmas aleatorias del mismo tamaño muestreadas de
  los genes medidos. Se reporta el percentil de |ρ_observado| en esa
  distribución. Una firma aleatoria en GBM no tiene ρ = 0 (la expresión está
  muy correlacionada), así que este control mide si la hipoxia destaca *sobre
  el fondo transcriptómico*, no frente a cero.
- Método de puntuación alternativo (singscore) y firma alternativa (Buffa).
- Exposición secundaria (volumen necrótico absoluto).
- Excluyendo pacientes con IDH desconocido.

### 4.3 Validación externa (CPTAC-GBM)
Mismo código, sin cambiar ningún parámetro:
- **V1 – ARN:** firma Hallmark sobre RNA-seq. Éxito: ρ > 0 con p unilateral < 0,05.
- **V2 – Proteína:** la misma firma puntuada sobre el proteoma (solo los genes
  de la firma cuantificados como proteína; se reporta la cobertura). Éxito
  igual que V1.
- Sensibilidad con pureza tumoral como covariable adicional.

**Interpretación de la validación (potencia limitada).** Con ~60 pacientes en
CPTAC, la correlación mínima detectable (unilateral, potencia 80 %) es
ρ ≈ 0,32: un efecto real de ρ ≈ 0,3 puede no alcanzar significación. Para no
confundir «sin potencia» con «sin efecto», cada capa de validación se
clasifica en tres niveles:

| Veredicto | Condición |
|---|---|
| **Replica** | ρ > 0 y p unilateral < 0,05 |
| **Consistente, potencia insuficiente** | No replica, pero ρ > 0 y el test de heterogeneidad con el descubrimiento no es significativo (z de Fisher, p ≥ 0,05) |
| **No replica** | ρ ≤ 0, o diferencia significativa con el descubrimiento |

Se reporta además la **estimación combinada** (efectos fijos sobre z de
Fisher, ponderando por n − 3 − k) de descubrimiento + validación en ARN, con
su IC 95 %. En la capa de proteína la comparación con el descubrimiento (ARN)
es solo orientativa, porque mide otra molécula.

### 4.4 Exploratorio (genera hipótesis, no conclusiones)
Asociación característica a característica (ρ parcial con las mismas
covariables, corrección de Benjamini-Hochberg dentro de cada capa, FDR 10 %):

| Capa | Pregunta exploratoria |
|---|---|
| Transcriptoma completo (TCGA y CPTAC) | ¿Qué otros programas acompañan a la necrosis? (GSEA preordenado sobre Hallmarks) |
| Proteoma | ¿Qué proteínas siguen a la necrosis, más allá de la firma? |
| Fosfoproteoma / acetiloma | ¿Hay señalización (p. ej. vías de estrés) asociada? |
| CNV y mutaciones | ¿Se asocia la necrosis a alteraciones driver (EGFR, PTEN, NF1…)? |
| Deconvolución inmune (xCell/CIBERSORT) | ¿Más necrosis implica más macrófagos? |
| miRNA | ¿Hay miRNAs asociados a hipoxia (p. ej. miR-210)? |
| Metaboloma / lipidoma (si se descargan) | ¿Se ve el giro glucolítico (lactato)? |

Todo lo de esta sección se reporta con la etiqueta «exploratorio».

## 5. Qué se reportará siempre (también si el resultado es negativo)
- n final tras cada criterio de exclusión (diagrama de flujo).
- ρ parcial, IC 95 % y p de cada análisis confirmatorio y de validación.
- Cobertura de la firma en cada plataforma.
- Gráfico de residuos (exposición vs resultado ajustados).
- Distribución del control negativo con el valor observado marcado.

## 6. Extensión (segmentación propia)
Repetir 4.1 sustituyendo las segmentaciones expertas por las de un segmentador
automático (YOLO). Se reporta la concordancia de la fracción necrótica
(correlación intraclase) y si el ρ parcial se mantiene dentro del IC original.

## 7. Implementación
`src/btc/radiogenomics/` (ver `btc rg-* --help`). Todo el pipeline se ejecuta
también sobre una cohorte sintética con efecto conocido (`scripts/demo_radiogenomics.py`)
para comprobar que los métodos recuperan el efecto plantado y no inventan uno
cuando no existe.

## 8. Registro de factibilidad (comprobaciones previas a los datos)

| Fecha | Comprobación | Resultado |
|---|---|---|
| 2026-10-02 | API de cBioPortal, estudio `gbm_tcga` | 528 muestras con U133A, 166 con RNA-seq, 290 secuenciadas, 15 mutaciones IDH1/2. El atributo de edad es `AGE`. Los endpoints usados por `btc.radiogenomics.cohorts.CBioPortal` responden como se espera. |
| 2026-10-02 | Acceso a imágenes en TCIA | Las colecciones cerebrales con DICOM original (TCGA-GBM, TCGA-LGG, CPTAC-GBM, REMBRANDT, Ivy GAP…) están bajo **TCIA Restricted License** (riesgo de reconstrucción facial): requieren cuenta de TCIA y acuerdo de licencia aprobado. De las colecciones cerebrales, solo UPENN-GBM es pública anónima. |
| 2026-10-02 | BraTS-TCGA-GBM | Paquete procesado (sin cráneo, con segmentaciones) de acceso abierto (CC BY 3.0) vía Aspera: **102 pacientes**. Es la fuente de imagen del descubrimiento; no requiere la licencia restringida. |
| pendiente | Solapamiento 102 ∩ U133A | Requiere la lista de pacientes del paquete BraTS-TCGA-GBM (`btc rg-overlap`). |
| 2026-10-02 | Licencia restringida de TCIA para CPTAC-GBM | **Concedida** (correo del Help Desk de TCIA, colección CPTAC-GBM, DOI 10.7937/K9/TCIA.2018.3RJE41Q1). El formulario de licencia firmado es de noviembre de 2024 y la RM se descargó en diciembre de 2024, así que la vigencia de 3 años probablemente cuenta desde entonces: **confirmar la fecha del correo de aprobación**. Los derivados por paciente siguen sin poder publicarse. |
| 2026-10-03 | Imagen de CPTAC-GBM en local | Encontrada en la carpeta local del TFM: descarga completa del NBIA Data Retriever (**~70 pacientes**: 71 elementos en la carpeta, DICOM original con `metadata.csv`), una descarga anterior incompleta (13 pacientes, descartar) y una conversión a NIfTI de todas las series (1.486 volúmenes, sin máscaras ni ID de paciente en el nombre, así que no se usa). En el paciente revisado hay FLAIR axial, T1 axial, T1 con contraste y T2. |
| pendiente | Imagen de BraTS-TCGA-GBM en local | No está en la carpeta del TFM, ni en Descargas ni en Documentos. Hay que descargarla (acceso abierto, unos 767 MB). |
| pendiente | Secuencias y segmentación de CPTAC-GBM | Contar a partir de `metadata.csv` cuántos de los ~70 tienen las 4 secuencias preoperatorias. Las máscaras no vienen con la colección: usar los casos CPTAC de BraTS 2021 si están, o una segmentación automática con un modelo BraTS preentrenado (que se declarará como desviación). |

## 9. Desviaciones del plan
| Fecha | Cambio | Motivo |
|---|---|---|
| — | — | — |
