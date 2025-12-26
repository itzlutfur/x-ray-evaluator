from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import tensorflow as tf


@dataclass(frozen=True)
class GradCamResult:
    heatmap: np.ndarray  # HxW float32 in [0,1]
    heatmap_png: bytes
    overlay_png: bytes
    layer_name: str


def find_last_conv_layer_name(model: tf.keras.Model) -> str:
    conv_like = (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D, tf.keras.layers.SeparableConv2D)

    # If the model contains a nested backbone (common for saved `.keras` checkpoints),
    # prefer the *last conv layer inside that backbone*.
    backbone, _ = _find_top_level_backbone(model)
    if backbone is not None:
        for layer in reversed(backbone.layers):
            if isinstance(layer, conv_like):
                return layer.name

    # Otherwise search the top-level graph.
    for layer in reversed(model.layers):
        if isinstance(layer, conv_like):
            return layer.name

    raise RuntimeError("Unable to find a convolutional feature layer for Grad-CAM.")


def _iter_layers_recursive(model: tf.keras.Model):
    for layer in model.layers:
        # Recurse into nested models
        if isinstance(layer, tf.keras.Model):
            yield from _iter_layers_recursive(layer)
        yield layer


def _get_layer_recursive(model: tf.keras.Model, name: str):
    # Try normal lookup first
    try:
        return model.get_layer(name)
    except Exception:
        pass
    for layer in model.layers:
        if layer.name == name:
            return layer
        if isinstance(layer, tf.keras.Model):
            try:
                return _get_layer_recursive(layer, name)
            except Exception:
                continue
    raise KeyError(name)


def compute_gradcam(
    *,
    model: tf.keras.Model,
    input_tensor: tf.Tensor,
    original_rgb: np.ndarray,
    last_conv_layer: str | None = None,
    class_index: int | None = None,
    colormap: int = cv2.COLORMAP_JET,
    alpha: float = 0.35,
) -> GradCamResult:
    if last_conv_layer is None:
        last_conv_layer = find_last_conv_layer_name(model)

    if input_tensor.ndim != 4:
        raise ValueError("input_tensor must be a 4D batch tensor.")

    h, w, _ = original_rgb.shape

    # First-choice path: if a nested backbone exists, use a Colab-style graph that is
    # stable across many saved checkpoints: grad_model(backbone.input) -> [activation, head].
    backbone, backbone_index = _find_top_level_backbone(model)
    if backbone is not None:
        grads, conv_outputs, used_layer_name = _gradcam_through_backbone(
            model=model,
            backbone=backbone,
            backbone_index=backbone_index,
            input_tensor=input_tensor,
            target_layer_name=last_conv_layer,
            class_index=class_index,
        )
        last_conv_layer = used_layer_name
    else:
        # Fallback path: simple models without nested backbone.
        grads, conv_outputs = _gradcam_through_top_level_layers(
            model=model,
            input_tensor=input_tensor,
            target_layer_name=last_conv_layer,
            class_index=class_index,
        )

    if grads is None:
        raise RuntimeError("Grad-CAM failed: gradients are None (layer mismatch).")

    # Global-average-pool gradients over spatial dims
    weights = tf.reduce_mean(grads, axis=(1, 2))  # (B, C)

    # Weighted sum of conv maps
    cam = tf.einsum("bhwc,bc->bhw", conv_outputs, weights)
    cam = tf.nn.relu(cam)

    cam_np = cam[0].numpy().astype(np.float32)
    if np.max(cam_np) > 0:
        cam_np = cam_np / (np.max(cam_np) + 1e-8)
    cam_np = cv2.resize(cam_np, (w, h), interpolation=cv2.INTER_LINEAR)
    cam_np = np.clip(cam_np, 0.0, 1.0)

    heatmap_u8 = (cam_np * 255.0).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_u8, colormap)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    overlay = (alpha * heatmap_color + (1.0 - alpha) * original_rgb).astype(np.uint8)

    heatmap_png = _encode_png(heatmap_color)
    overlay_png = _encode_png(overlay)

    return GradCamResult(
        heatmap=cam_np,
        heatmap_png=heatmap_png,
        overlay_png=overlay_png,
        layer_name=last_conv_layer,
    )


def _select_score(preds: tf.Tensor, class_index: int | None) -> tf.Tensor:
    if isinstance(preds, (list, tuple)) and len(preds) == 1:
        preds = preds[0]
    if not isinstance(preds, tf.Tensor):
        preds = tf.convert_to_tensor(preds)

    # Binary sigmoid head: replicate your Colab logic to also produce meaningful maps
    # for non-fracture predictions.
    if preds.shape.rank == 2 and preds.shape[-1] == 1:
        p = preds[:, 0]
        if class_index is None:
            return tf.where(p >= 0.5, p, 1.0 - p)
        return p if int(class_index) == 1 else (1.0 - p)

    if preds.shape.rank == 2 and preds.shape[-1] == 2:
        idx = 1 if class_index is None else int(class_index)
        return preds[:, idx]

    return tf.reduce_max(preds, axis=-1)


def _find_top_level_backbone(model: tf.keras.Model) -> tuple[tf.keras.Model | None, int]:
    # Heuristic: choose the last top-level nested Model/Functional producing a 4D feature map.
    for idx in range(len(model.layers) - 1, -1, -1):
        layer = model.layers[idx]
        if not isinstance(layer, tf.keras.Model):
            continue
        try:
            out_shape = layer.output_shape
        except Exception:
            continue
        if isinstance(out_shape, tuple) and len(out_shape) == 4:
            return layer, idx
    return None, -1


def _gradcam_through_backbone(
    *,
    model: tf.keras.Model,
    backbone: tf.keras.Model,
    backbone_index: int,
    input_tensor: tf.Tensor,
    target_layer_name: str,
    class_index: int | None,
) -> tuple[tf.Tensor, tf.Tensor, str]:
    # Build a clean, connected graph from `backbone.input` to avoid ambiguous `.output`
    # when the backbone layer has multiple inbound nodes.
    inp = backbone.input
    try:
        backbone_out = backbone(inp, training=False)
    except TypeError:
        backbone_out = backbone(inp)

    if isinstance(backbone_out, (list, tuple)):
        if len(backbone_out) == 0:
            raise RuntimeError("Backbone produced no outputs.")
        backbone_out = backbone_out[0]

    # Choose activation inside backbone if possible, else use backbone output.
    used_name = target_layer_name
    try:
        activation_tensor = backbone.get_layer(target_layer_name).output
    except Exception:
        activation_tensor = backbone_out
        used_name = backbone.name

    if isinstance(activation_tensor, (list, tuple)) and len(activation_tensor) == 1:
        activation_tensor = activation_tensor[0]

    # Build classifier head from backbone output, using remaining top-level layers.
    x = backbone_out
    for layer in model.layers[backbone_index + 1 :]:
        if isinstance(layer, tf.keras.layers.InputLayer):
            continue
        try:
            x = layer(x, training=False)
        except TypeError:
            x = layer(x)

    head_output = x
    if isinstance(head_output, (list, tuple)) and len(head_output) == 1:
        head_output = head_output[0]

    def run_with_activation(act_tensor: tf.Tensor, name: str):
        gm = tf.keras.Model(inputs=inp, outputs=[act_tensor, head_output])
        with tf.GradientTape() as tape:
            conv, preds = gm(input_tensor)
            if isinstance(conv, (list, tuple)) and len(conv) == 1:
                conv = conv[0]
            tape.watch(conv)
            score = _select_score(preds, class_index)
        g = tape.gradient(score, conv)
        return g, conv, name

    grads, conv_outputs, used_name = run_with_activation(activation_tensor, used_name)

    # If the internal activation isn't on the classification path (e.g., multi-output backbones),
    # fall back to the backbone output feature map, which is guaranteed to feed the head.
    if grads is None:
        grads, conv_outputs, used_name = run_with_activation(backbone_out, backbone.name)

    return grads, conv_outputs, used_name


def _gradcam_through_top_level_layers(
    *,
    model: tf.keras.Model,
    input_tensor: tf.Tensor,
    target_layer_name: str,
    class_index: int | None,
) -> tuple[tf.Tensor, tf.Tensor]:
    with tf.GradientTape() as tape:
        x = input_tensor
        activation = None
        for layer in model.layers:
            if isinstance(layer, tf.keras.layers.InputLayer):
                continue
            try:
                x = layer(x, training=False)
            except TypeError:
                x = layer(x)
            if layer.name == target_layer_name:
                activation = x
                tape.watch(activation)

        if activation is None:
            raise RuntimeError(f"Grad-CAM target layer not reached: {target_layer_name}")

        score = _select_score(x, class_index)
    grads = tape.gradient(score, activation)
    return grads, activation


 


def _encode_png(rgb: np.ndarray) -> bytes:
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".png", bgr)
    if not ok:
        raise RuntimeError("Failed to encode PNG.")
    return buf.tobytes()
