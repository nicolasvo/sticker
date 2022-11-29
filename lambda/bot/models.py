import cv2
import numpy as np
from rembg import remove

from image import add_outline, rescale_img, write_img


def segment_u2net(img_path, out_path):
    with open(img_path, "rb") as i:
        with open(out_path, "wb") as o:
            input = i.read()
            output = remove(input, alpha_matting=True, alpha_matting_erode_size=0)
            o.write(output)
    img = cv2.imread(out_path, cv2.IMREAD_UNCHANGED)
    img = add_outline(img, threshold=0, stroke_size=6, colors=((255, 255, 255),))
    scaled_img = rescale_img(np.array(img))
    write_img(scaled_img, out_path, alpha=True)


def segment_modnet(img_path, out_path, t=200):
    # load image
    frame = cv2.imread(img_path)
    blob = cv2.resize(frame, (672, 512), cv2.INTER_AREA)
    blob = blob.astype(np.float)
    blob /= 255
    blob = 2 * blob - 1
    channels = cv2.split(blob)
    blob = np.array([[channels[2], channels[1], channels[0]]])

    # load model
    model_path = "modnet_photographic_portrait_matting_opset9.onnx"
    net = cv2.dnn.readNetFromONNX(model_path)

    # select CPU
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    # Sets the input to the network
    net.setInput(blob)

    # Runs the forward pass to get output of the output layers
    outs = net.forward()

    # Process the result
    mask = outs[0][0]
    mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
    mask = cv2.merge([mask, mask, mask])
    result = (mask * frame + (1 - mask) * np.ones_like(frame) * 255).astype(np.uint8)

    height, width, _ = result.shape
    n_channels = 4

    m = (mask * 255).astype(np.uint8)
    unique = np.unique(m)
    f = np.full((height, width, 3), 155)
    m[np.where(m < t)] = 0
    m[np.where(m >= t)] = 1

    scaled_img = cv2.cvtColor(result, cv2.COLOR_BGR2BGRA)
    m = cv2.cvtColor(m, cv2.COLOR_BGR2BGRA)

    transparent_img = np.zeros((height, width, n_channels), dtype=np.uint8)

    out = np.where(m, scaled_img, transparent_img)

    alpha = out[:, :, 3]
    alpha[np.all(out[:, :, 0:3] == (0, 0, 0), 2)] = 0

    scaled_img = rescale_img(out)
    cv2.imwrite(out_path, scaled_img)
    img = cv2.imread(out_path, cv2.IMREAD_UNCHANGED)
    img = add_outline(img, threshold=0, stroke_size=6, colors=((255, 255, 255),))
    write_img(np.array(img), out_path, alpha=True)
