# Datos y licencias: qué se puede publicar

Este repositorio publica **código y resultados agregados**. Nunca datos de
pacientes. Este documento resume, fuente por fuente, qué permite cada licencia.

> No es asesoramiento legal: es una lectura de las licencias públicas de cada
> fuente. Ante una publicación formal, revisa el texto de la licencia que
> aceptaste al descargar los datos.

## Resumen

| Fuente | Contenido | Licencia | ¿Datos en el repo? | ¿Resultados por paciente publicables? |
|---|---|---|---|---|
| BraTS-TCGA-GBM (TCIA) | RM sin cráneo + segmentaciones | CC BY 3.0 (abierta) | No (se descargan aparte) | Sí, citando la fuente |
| cBioPortal / GDC — TCGA-GBM | Microarray, mutaciones, clínica (nivel abierto) | Acceso abierto | No | Sí, citando la fuente |
| PDC / GDC / `cptac` — CPTAC-GBM | ARN, proteoma, fosfoproteoma… (nivel abierto) | Acceso abierto | No | Sí, citando la fuente |
| **CPTAC-GBM (TCIA), RM original** | DICOM con rasgos faciales | **TCIA Restricted License** | **No** | **No**: solo resultados agregados |
| UPENN-GBM (TCIA) | RM sin cráneo + segmentaciones + clínica | Pública en TCIA | No | Sí, citando la fuente |
| Kaggle Brain Tumor Classification | Imágenes 2D | Licencia del dataset en Kaggle | No | Revisar la licencia del dataset |

## La licencia restringida de TCIA, en la práctica

Desde 2022, las colecciones de cerebro y cabeza y cuello con imagen original
(TCGA-GBM, TCGA-LGG, CPTAC-GBM, REMBRANDT, Ivy GAP…) están bajo la *TCIA
Restricted License*, porque de una RM craneal se podría reconstruir la cara del
paciente. Requiere cuenta de TCIA y un acuerdo aprobado. Sus puntos clave:

- ✅ **Publicar resultados** de la investigación está permitido y se fomenta.
- ❌ **Redistribuir** las imágenes, o compartirlas con quien no figure en el acuerdo.
- ❌ **Obras derivadas** (máscaras, mapas, tablas de características por
  paciente) heredan las mismas restricciones que las imágenes: no se publican.
- ❌ Uso comercial, reidentificación o reconstrucción facial.
- 📌 Agradecer a TCIA y la financiación del NIH, citar la colección en el formato
  que indica TCIA y enviar copia de cortesía de los manuscritos publicados.
- 📌 La licencia cubre el proyecto descrito al solicitarla y dura 3 años. Si se
  solicitó para el TFM, confirmar que este estudio entra en ese alcance.

**Estado en este proyecto:** acceso a CPTAC-GBM **concedido** por TCIA. El
formulario se firmó en noviembre de 2024; la vigencia de 3 años cuenta desde la
aprobación (comprobar la fecha en el correo). Solo cubre al titular de la cuenta:
un coautor necesitaría su propia aprobación.

## Cómo lo aplica este repositorio

1. **Los datos nunca entran en git.** `.gitignore` excluye `data/`, `runs/`,
   NIfTI y DICOM.
2. **Cada cohorte declara su licencia** en el YAML del estudio
   (`license: open` o `license: restricted`).
3. **`btc rg-analyze` separa las salidas.** Las tablas por paciente de una cohorte
   restringida se escriben en `<out-dir>/solo_local/`, que también está en
   `.gitignore`. Lo publicable (informe con estimaciones agregadas, figuras de
   la cohorte abierta, tablas exploratorias por característica) queda fuera de
   esa carpeta.
4. **Comprobación antes de publicar.** `btc rg-check-publish --dir <carpeta>`
   busca identificadores de pacientes de cohortes restringidas (p. ej.
   `C3L-00016`) en cualquier fichero de texto fuera de `solo_local/`. El análisis
   la ejecuta automáticamente al terminar y avisa si encuentra alguno.

## Reproducibilidad

- **Descubrimiento (TCGA):** reproducible por cualquiera, todas las fuentes son
  abiertas.
- **Validación (CPTAC):** reproducible por cualquiera que obtenga su propia
  licencia de TCIA, con el mismo código y configuración.

## Citas
- BraTS-TCGA-GBM: Bakas et al., *Scientific Data* 4:170117 (2017), y el DOI de TCIA de la colección.
- TCGA-GBM: según las directrices de publicación de TCGA y cBioPortal (Cerami et al. 2012; Gao et al. 2013).
- CPTAC-GBM: Wang et al., *Cancer Cell* 39:509 (2021), y el DOI de TCIA de la colección.
- MSigDB Hallmarks: Liberzon et al., *Cell Systems* 1:417 (2015).
