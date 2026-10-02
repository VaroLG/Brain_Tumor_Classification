"""Pipeline de datos con ``tf.data`` a partir del manifest.

Sustituye a ``ImageDataGenerator.flow_from_directory`` (obsoleto en Keras 3).

El fallo corregido: aumento de datos en validación y test
---------------------------------------------------------
En los notebooks se creaba UN solo ``ImageDataGenerator`` con rotaciones,
desplazamientos, zoom y volteos, y se usaba para train, val **y test**. Eso
significa que:
- cada evaluación veía imágenes deformadas al azar distintas -> la métrica
  cambiaba entre ejecuciones sin tocar el modelo;
- se evaluaba sobre imágenes que no son las reales del paciente.
El aumento de datos es una técnica de *entrenamiento*. Aquí solo se aplica
cuando ``training=True``.

Además, las imágenes se leen desde la columna ``image_path`` del manifest, así
que ya no hace falta copiar físicamente cada imagen a carpetas train/val/test
(el original duplicaba todo el dataset en disco con ``shutil.copy``).
"""

from __future__ import annotations

import keras
import pandas as pd
import tensorflow as tf

from btc.models import get_backbone

AUTOTUNE = tf.data.AUTOTUNE


def make_augmenter(seed: int = 42) -> keras.Sequential:
    """Mismas transformaciones que el TFM (rotación ±20º, desplazamiento y zoom 10 %,
    volteo horizontal), ahora como capas de Keras que corren en el grafo de TF.

    El volteo horizontal es razonable porque el cerebro es aproximadamente
    simétrico izquierda-derecha y la lateralidad del tumor no define la clase.
    """
    return keras.Sequential(
        [
            keras.layers.RandomFlip("horizontal", seed=seed),
            keras.layers.RandomRotation(20 / 360, fill_mode="constant", seed=seed),
            keras.layers.RandomTranslation(0.1, 0.1, fill_mode="constant", seed=seed),
            keras.layers.RandomZoom(0.1, fill_mode="constant", seed=seed),
        ],
        name="augmenter",
    )


def _load_image(path: tf.Tensor, image_size: int) -> tf.Tensor:
    """Lee PNG/JPG, fuerza 1 canal y lo replica a 3 (las redes de ImageNet esperan RGB).

    Las RM son en escala de grises: replicar el canal no añade información, solo
    adapta la forma de entrada a la de los pesos preentrenados.
    """
    raw = tf.io.read_file(path)
    img = tf.io.decode_image(raw, channels=1, expand_animations=False)
    img = tf.image.resize(img, (image_size, image_size), antialias=True)
    return tf.image.grayscale_to_rgb(img)  # float32 en rango 0-255


def make_dataset(
    df: pd.DataFrame,
    classes: list[str],
    backbone: str,
    image_size: int = 240,
    batch_size: int = 32,
    training: bool = False,
    augment: bool = True,
    seed: int = 42,
) -> tf.data.Dataset:
    """Crea un ``tf.data.Dataset`` de pares (imagen preprocesada, índice de clase).

    El orden de ``classes`` define el índice de cada clase y se guarda en la
    config del run: el original dependía del orden alfabético de las carpetas,
    lo que obligaba a recordar a mano que "clase 0" era "HGG", etc.
    """
    class_to_idx = {c: i for i, c in enumerate(classes)}
    unknown = set(df["label"]) - set(class_to_idx)
    if unknown:
        raise ValueError(f"Etiquetas en el manifest que no están en 'classes': {unknown}")

    paths = df["image_path"].astype(str).to_numpy()
    labels = df["label"].map(class_to_idx).astype("int32").to_numpy()
    preprocess = get_backbone(backbone).preprocess

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(len(df), seed=seed, reshuffle_each_iteration=True)
    ds = ds.map(lambda p, y: (_load_image(p, image_size), y), num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size)
    if training and augment:
        aug = make_augmenter(seed)
        ds = ds.map(lambda x, y: (aug(x, training=True), y), num_parallel_calls=AUTOTUNE)
    # Normalización específica del backbone (siempre, en train y en eval)
    ds = ds.map(lambda x, y: (preprocess(x), y), num_parallel_calls=AUTOTUNE)
    return ds.prefetch(AUTOTUNE)
