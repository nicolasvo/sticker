import tempfile

import cv2
import numpy as np
from PIL import Image

from lang_sam import LangSAM
from lang_sam.utils import draw_image


def segment_photo(
    image_path, text_prompt=None
) -> None:  # TODO: if no text prompt, segment everything
    with tempfile.TemporaryDirectory(dir="/tmp/") as tmpdirname:
        masks, boxes, _, _ = segment(image_path, text_prompt, tmpdirname)
        postprocessing(imgae_path, masks, boxes, tmpdirname, white_outline=False)
        output_path = f"{tmpdirname}/output.png"  # TODO: dev ugly


def segment(photo_path, text_prompt, tmpdirname):
    model = LangSAM()
    image_pil = Image.open(photo_path).convert("RGB")
    output_path = f"{tmpdirname}/output_draw.png"
    masks, boxes, phrases, logits = model.predict(image_pil, text_prompt)
    labels = [f"{phrase} {logit:.2f}" for phrase, logit in zip(phrases, logits)]
    image_array = np.asarray(image_pil)
    image_draw = image = draw_image(image_array, masks, boxes, labels)
    image = Image.fromarray(np.uint8(image)).convert("RGB")
    cv2.imwrite(output_path, cv2.cvtColor(image_draw, cv2.COLOR_RGBA2BGRA))

    return masks, boxes, phrases, logits
