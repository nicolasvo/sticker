import cv2
import numpy as np
from rembg import remove

from image import add_outline, rescale_img, write_img


def segment_u2net(img_path, out_path):
    with open(img_path, "rb") as i:
        with open(out_path, "wb") as o:
            input = i.read()
            output = remove(input)
            o.write(output)
    img = cv2.imread(out_path, cv2.IMREAD_UNCHANGED)
    img = add_outline(img, threshold=0, stroke_size=6, colors=((255, 255, 255),))
    scaled_img = rescale_img(np.array(img))
    write_img(scaled_img, out_path, alpha=True)
