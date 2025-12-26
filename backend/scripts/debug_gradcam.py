from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

import numpy as np
import tensorflow as tf

# Allow running as: python scripts/debug_gradcam.py ...
BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.services.model_registry import get_or_load_model  # noqa: E402
from app.utils.gradcam import compute_gradcam, find_last_conv_layer_name  # noqa: E402
from app.utils.image_io import decode_image_bytes  # noqa: E402
from app.utils.preprocessing import preprocess_for_model  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=str)
    parser.add_argument("--model", type=str, default="ResNet50")
    parser.add_argument("--layer", type=str, default=None, help="Optional layer name override")
    args = parser.parse_args()

    img_path = Path(args.image)
    data = img_path.read_bytes()
    decoded = decode_image_bytes(data)

    loaded = get_or_load_model(args.model)
    model = loaded.model

    print("loaded:", loaded.path)
    print("model:", model.name)
    print("inputs:", model.inputs)
    print("output:", model.output)

    x = preprocess_for_model(decoded.rgb)
    batch = tf.convert_to_tensor(np.expand_dims(x, 0), dtype=tf.float32)

    last = args.layer or find_last_conv_layer_name(model)
    print("last_conv_candidate:", last)

    try:
        gc = compute_gradcam(model=model, input_tensor=batch, original_rgb=decoded.rgb, last_conv_layer=last)
        print("gradcam_ok:", gc.layer_name, "heatmap", gc.heatmap.shape)
    except Exception as e:
        print("gradcam_failed:", type(e).__name__, str(e))
        traceback.print_exc()
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
