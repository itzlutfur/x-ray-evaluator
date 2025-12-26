from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

import numpy as np
import tensorflow as tf

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.services.model_registry import get_or_load_model  # noqa: E402
from app.utils.gradcam import compute_gradcam, find_last_conv_layer_name  # noqa: E402
from app.utils.image_io import decode_image_bytes  # noqa: E402
from app.utils.preprocessing import preprocess_for_model  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument("--model", default="ResNet50")
    parser.add_argument("--layer", default=None)
    args = parser.parse_args()

    img_path = Path(args.image)
    decoded = decode_image_bytes(img_path.read_bytes())

    loaded = get_or_load_model(args.model)
    model = loaded.model

    x = preprocess_for_model(decoded.rgb)
    batch = tf.convert_to_tensor(np.expand_dims(x, 0), dtype=tf.float32)

    layer = args.layer or find_last_conv_layer_name(model)
    print("target layer:", layer)

    try:
        gc = compute_gradcam(model=model, input_tensor=batch, original_rgb=decoded.rgb, last_conv_layer=layer)
        print("OK layer:", gc.layer_name, "heatmap", gc.heatmap.shape, "png sizes", len(gc.heatmap_png), len(gc.overlay_png))
        return 0
    except Exception as e:
        print("FAILED:", type(e).__name__, str(e))
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
