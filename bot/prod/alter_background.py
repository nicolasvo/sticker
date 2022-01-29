import numpy as np
import cv2
from pixellib.tune_bg import alter_bg


class AlterBackground(alter_bg):
    def remove_bg(
        self,
        img_path,
        detect=None,
    ):

        scaled_img_path = (
            img_path.replace(img_path.split("/")[-1], "")
            + f'scaled_{img_path.split("/")[-1]}'
        )

        ori_img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        scaled_img = self.rescale_img(ori_img)
        self.write_img(scaled_img, scaled_img_path)

        seg_image = self.segmentAsPascalvoc(scaled_img_path)

        if detect is not None:
            target_class = self.target_obj(detect)
            seg_image[1][seg_image[1] != target_class] = 0

        scaled_img = cv2.cvtColor(scaled_img, cv2.COLOR_BGR2BGRA)
        seg_img = cv2.cvtColor(seg_image[1], cv2.COLOR_BGR2BGRA)
        height, width, _ = scaled_img.shape

        n_channels = 4
        transparent_img = np.zeros((height, width, n_channels), dtype=np.uint8)

        out = np.where(seg_img, scaled_img, transparent_img)
        alpha = out[:, :, 3]
        alpha[np.all(out[:, :, 0:3] == (0, 0, 0), 2)] = 0

        return out

    def rescale_img(self, img):
        height, width, _ = img.shape
        if [height, width].index(max([height, width])) == 0:
            factor = 512 / height
            height = 512
            width = int(width * factor)
        else:
            factor = 512 / width
            width = 512
            height = int(height * factor)

        img = cv2.resize(
            img, dsize=[width, height], dst=factor, interpolation=cv2.INTER_LINEAR
        )

        return img

    def write_img(self, img, out_img_path):
        cv2.imwrite(out_img_path, img)

    def boom(self, img_path, out_img_path, detect):
        img = self.remove_bg(img_path, detect)
        self.write_img(img, out_img_path)