import numpy as np
import cv2
from PIL import Image
from pixellib.tune_bg import alter_bg
from image import rescale_img, write_img


class AlterBackground(alter_bg):
    def remove_bg(self, img_path):
        scaled_img_path = (
            img_path.replace(img_path.split("/")[-1], "")
            + f'scaled_{img_path.split("/")[-1]}'
        )

        ori_img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        scaled_img = rescale_img(ori_img)
        write_img(scaled_img, scaled_img_path)

        seg_image = self.segmentAsPascalvoc(scaled_img_path)

        cat = self.target_obj("cat")[::-1]
        dog = self.target_obj("dog")[::-1]
        person = self.target_obj("person")[::-1]
        mask_remove = (
            np.any(seg_image[1] != cat, axis=-1)
            & (np.any(seg_image[1] != dog, axis=-1))
            & (np.any(seg_image[1] != person, axis=-1))
        )

        seg_image[1][mask_remove] = [0, 0, 0]
        seg_image[1][~mask_remove] = [255, 255, 255]

        scaled_img = cv2.cvtColor(scaled_img, cv2.COLOR_BGR2BGRA)
        seg_img = cv2.cvtColor(seg_image[1], cv2.COLOR_BGR2BGRA)
        height, width, _ = scaled_img.shape

        n_channels = 4
        transparent_img = np.zeros((height, width, n_channels), dtype=np.uint8)

        out = np.where(seg_img, scaled_img, transparent_img)
        alpha = out[:, :, 3]
        alpha[np.all(out[:, :, 0:3] == (0, 0, 0), 2)] = 0

        return out

    def segmentAsPascalvoc(self, image_path, process_frame=False):
        if process_frame == True:
            image = image_path
        else:
            image = cv2.imread(image_path)

        h, w, n = image.shape

        if n > 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        resize_ratio = 1.0 * self.INPUT_SIZE / max(w, h)
        target_size = (int(resize_ratio * w), int(resize_ratio * h))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resized_image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
        batch_seg_map = self.sess.run(
            self.OUTPUT_TENSOR_NAME,
            feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]},
        )

        seg_image = batch_seg_map[0]
        raw_labels = seg_image
        labels = self.obtain_segmentation(seg_image)
        labels = np.array(
            Image.fromarray(labels.astype("uint8")).resize(
                (w, h), resample=Image.NEAREST
            )
        )
        labels = cv2.cvtColor(labels, cv2.COLOR_RGB2BGR)

        return raw_labels, labels

    def obtain_segmentation(self, image, nc=21):
        colors = self.label_pascal()
        r = np.zeros_like(image).astype(np.uint8)
        g = np.zeros_like(image).astype(np.uint8)
        b = np.zeros_like(image).astype(np.uint8)

        for a in range(0, nc):
            index = image == a
            r[index] = colors[a, 0]
            g[index] = colors[a, 1]
            b[index] = colors[a, 2]
            rgb = np.stack([r, g, b], axis=2)

        return rgb

    def label_pascal(self):
        return np.asarray(
            [
                [0, 0, 0],
                [128, 0, 0],
                [0, 128, 0],
                [255, 255, 255],
                [0, 0, 128],
                [128, 0, 128],
                [0, 128, 128],
                [128, 128, 128],
                [64, 0, 0],
                [192, 0, 0],
                [64, 128, 0],
                [192, 128, 0],
                [64, 0, 128],
                [192, 0, 128],
                [64, 128, 128],
                [255, 255, 255],
                [0, 64, 0],
                [128, 64, 0],
                [0, 192, 0],
                [128, 192, 0],
                [0, 64, 128],
            ]
        )

    def boom(self, img_path, out_img_path):
        img = self.remove_bg(img_path)
        write_img(img, out_img_path)
