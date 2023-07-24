import cv2
import numpy as np
from PIL import Image

from lang_sam import LangSAM
from lang_sam.utils import draw_image

output_path = "/tmp/output.png"
image_path = "/tmp/input.jpeg"


def segment(image_path, text_prompt):
    # TODO: if no text prompt, segment everything
    masks, boxes, phrases, logits = _segment(image_path, text_prompt)
    return masks, boxes, phrases, logits


def _segment(photo_path, text_prompt):
    model = LangSAM()
    image_pil = Image.open(photo_path).convert("RGB")
    masks, boxes, phrases, logits = model.predict(image_pil, text_prompt)
    labels = [f"{phrase} {logit:.2f}" for phrase, logit in zip(phrases, logits)]
    image_array = np.asarray(image_pil)
    image_draw = image = draw_image(image_array, masks, boxes, labels)
    image = Image.fromarray(np.uint8(image)).convert("RGB")
    cv2.imwrite(output_path, cv2.cvtColor(image_draw, cv2.COLOR_RGBA2BGRA))

    return masks, boxes, phrases, logits
