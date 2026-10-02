"""btc — Brain Tumor Classification.

Paquete reutilizable que sustituye a los notebooks del TFM (ver ``legacy/``).

El flujo completo se articula alrededor de un único "contrato de datos":
el **manifest**, un CSV con una fila por imagen 2D y, como mínimo, las columnas

    image_path   ruta al PNG
    patient_id   identificador del paciente (o grupo) del que procede el corte
    label        clase (texto)

Todos los pasos leen o escriben un manifest:

    construir manifest  ->  split por paciente  ->  entrenar  ->  evaluar

Tener ``patient_id`` en cada fila es lo que permite evitar la fuga de datos
(data leakage) que tenían los notebooks originales.
"""

__version__ = "0.2.0"
