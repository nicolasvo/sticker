import cv2


def rescale_img(img):
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


def load_img(img_path):
    return cv2.imread(img_path)


def write_img(img, out_img_path):
    cv2.imwrite(out_img_path, img)
