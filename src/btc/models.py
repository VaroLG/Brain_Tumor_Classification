"""Modelos de transfer learning.

Cambios respecto a los notebooks:

1. **Preprocesado correcto para cada backbone.** Cada red de ImageNet se
   entrenó con una normalización concreta y espera recibir la misma:
     - VGG19 / ResNet50: "caffe" (BGR y resta de la media de ImageNet), rango 0-255
     - MobileNetV2 / Xception: escalado a [-1, 1]
     - EfficientNet: recibe 0-255 y normaliza internamente
   El notebook de supervivencia pasaba a VGG19 imágenes en [0, 1]
   (``rescale=1/255``): las activaciones de la primera capa quedaban en un rango
   para el que los filtros preentrenados no estaban calibrados, y se pierde
   buena parte de la ventaja del transfer learning.

2. **Cabeza pequeña y regularizada.** La original apilaba Dense(2560) ->
   Dense(1280) -> Dense(1024): ~6 millones de parámetros nuevos entrenados
   desde cero con unos cientos de pacientes. Es la receta clásica del
   sobreajuste que se ve en las curvas del TFM (train ~0.95, val ~0.65).
   Aquí: GlobalAveragePooling -> Dropout -> Dense(256) -> Dropout -> salida.

3. **El BatchNormalization del backbone se mantiene congelado** incluso en
   fine-tuning (``training=False``). Con lotes pequeños, actualizar sus
   estadísticas destruye lo aprendido en ImageNet.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import keras
from keras import layers
from keras.applications import (
    efficientnet,
    mobilenet_v2,
    resnet50,
    vgg19,
    xception,
)


@dataclass(frozen=True)
class Backbone:
    constructor: Callable[..., keras.Model]
    preprocess: Callable  # se aplica en el pipeline de datos, sobre imágenes RGB 0-255


BACKBONES: dict[str, Backbone] = {
    "efficientnetb0": Backbone(efficientnet.EfficientNetB0, efficientnet.preprocess_input),
    "efficientnetb1": Backbone(efficientnet.EfficientNetB1, efficientnet.preprocess_input),
    "vgg19": Backbone(vgg19.VGG19, vgg19.preprocess_input),
    "resnet50": Backbone(resnet50.ResNet50, resnet50.preprocess_input),
    "mobilenetv2": Backbone(mobilenet_v2.MobileNetV2, mobilenet_v2.preprocess_input),
    "xception": Backbone(xception.Xception, xception.preprocess_input),
}


def get_backbone(name: str) -> Backbone:
    try:
        return BACKBONES[name.lower()]
    except KeyError as err:
        raise ValueError(f"Backbone '{name}' no soportado. Opciones: {sorted(BACKBONES)}") from err


def build_model(
    backbone: str,
    n_classes: int,
    image_size: int = 240,
    dense_units: list[int] | None = None,
    dropout: float = 0.3,
    weights: str | None = "imagenet",
) -> tuple[keras.Model, keras.Model]:
    """Construye ``(modelo_completo, base_convolucional)``.

    Se devuelve también la base para poder descongelar capas en la fase 2.
    La salida es siempre softmax con ``n_classes`` neuronas (también en
    binario), así la misma función de pérdida y las mismas métricas sirven
    para todas las tareas.
    """
    dense_units = [256] if dense_units is None else dense_units
    base = get_backbone(backbone).constructor(
        include_top=False, weights=weights, input_shape=(image_size, image_size, 3)
    )
    pretrained = weights is not None
    # Con pesos preentrenados la base empieza congelada (fase 1). Sin ellos no
    # hay nada que proteger: una base aleatoria congelada solo produce ruido, así
    # que se entrena entera desde el principio (y no hay fase 2).
    base.trainable = not pretrained

    inputs = keras.Input(shape=(image_size, image_size, 3), name="image")
    # Preentrenada: training=False -> su BatchNorm siempre en modo inferencia.
    # Aleatoria: training=None -> BatchNorm aprende sus estadísticas normalmente.
    x = base(inputs, training=False if pretrained else None)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(dropout, name="dropout_gap")(x)
    for i, units in enumerate(dense_units):
        x = layers.Dense(units, activation="relu", name=f"dense_{i}")(x)
        x = layers.Dropout(dropout, name=f"dropout_{i}")(x)
    outputs = layers.Dense(n_classes, activation="softmax", name="probs")(x)
    return keras.Model(inputs, outputs, name=f"{backbone}_classifier"), base


def unfreeze_top_layers(base: keras.Model, n_layers: int) -> int:
    """Descongela las ``n_layers`` últimas capas de la base salvo las BatchNormalization.

    Devuelve cuántas capas quedaron entrenables (para el log).
    """
    base.trainable = True
    cutoff = max(len(base.layers) - n_layers, 0)
    for i, layer in enumerate(base.layers):
        layer.trainable = i >= cutoff and not isinstance(layer, layers.BatchNormalization)
    return sum(layer.trainable for layer in base.layers)
