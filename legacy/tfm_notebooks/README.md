# Notebooks originales del TFM

Código tal como se entregó con el Trabajo Fin de Máster (enero de 2025). Se
conserva sin modificar para que los resultados de la memoria
([`docs/memoria_tfm.pdf`](../../docs/memoria_tfm.pdf)) sigan siendo trazables.

**No usar como punto de partida**: contiene problemas metodológicos (split por
imagen, aumento de datos en test, preprocesado incorrecto para VGG19…)
descritos en [`docs/METODOLOGIA.md`](../../docs/METODOLOGIA.md) y corregidos en
el paquete `src/btc/`.

| Carpeta | Contenido | Sustituido por |
|---|---|---|
| `subtipos_kaggle/` | Clasificación de 4 subtipos con EfficientNetB1 | `btc manifest-folders` + `configs/subtypes_kaggle.yaml` |
| `grado_glioma_rembrandt/` | DICOM→PNG, recorte, split, grado II/III/IV y LGG/HGG | `btc manifest-dicom` + `configs/glioma_*` |
| `glioblastoma_upenn/` | NIfTI→PNG, split SG/MGMT/IDH1, VGG19 para SG | `btc manifest-upenn` + `configs/gbm_*` |
