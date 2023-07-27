import base64

import boto3
import cv2
import numpy as np
from PIL import Image


def upload_file_and_get_presigned_url(bucket_name, file_key, local_file_path):
    # Create a Boto3 client for S3
    s3_client = boto3.client("s3")

    # Upload the file to the S3 bucket
    s3_client.upload_file(local_file_path, bucket_name, file_key)

    # Generate a presigned URL for the uploaded file
    presigned_url = s3_client.generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket_name, "Key": file_key},
        ExpiresIn=3600,  # URL expiration time in seconds (e.g., 1 hour)
    )

    print(f"presigned url: {presigned_url}")

    return presigned_url


def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read())
        return encoded_image.decode("utf-8")


def base64_to_image(base64_string, output_file_path):
    with open(output_file_path, "wb") as image_file:
        decoded_image = base64.b64decode(base64_string)
        image_file.write(decoded_image)


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


def write_img(img, out_img_path, alpha=False):
    if alpha:
        cv2.imwrite(out_img_path, cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA))
    else:
        cv2.imwrite(out_img_path, img)


def add_outline(origin_image, threshold, stroke_size, colors):
    img = np.array(origin_image)
    h, w, _ = img.shape
    padding = stroke_size
    alpha = img[:, :, 3]
    rgb_img = img[:, :, 0:3]
    bigger_img = cv2.copyMakeBorder(
        rgb_img,
        padding,
        padding,
        padding,
        padding,
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0, 0),
    )
    alpha = cv2.copyMakeBorder(
        alpha, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=0
    )
    bigger_img = cv2.merge((bigger_img, alpha))
    h, w, _ = bigger_img.shape

    _, alpha_without_shadow = cv2.threshold(
        alpha, threshold, 255, cv2.THRESH_BINARY
    )  # threshold=0 in photoshop
    alpha_without_shadow = 255 - alpha_without_shadow
    dist = cv2.distanceTransform(
        alpha_without_shadow, cv2.DIST_L2, cv2.DIST_MASK_3
    )  # dist l1 : L1 , dist l2 : l2
    stroked = change_matrix(dist, stroke_size)
    stroke_alpha = (stroked * 255).astype(np.uint8)

    stroke_b = np.full((h, w), colors[0][2], np.uint8)
    stroke_g = np.full((h, w), colors[0][1], np.uint8)
    stroke_r = np.full((h, w), colors[0][0], np.uint8)

    stroke = cv2.merge((stroke_b, stroke_g, stroke_r, stroke_alpha))
    stroke = cv2pil(stroke)
    bigger_img = cv2pil(bigger_img)
    result = Image.alpha_composite(stroke, bigger_img)
    return result


def change_matrix(input_mat, stroke_size):
    stroke_size = stroke_size - 1
    mat = np.ones(input_mat.shape)
    check_size = stroke_size + 1.0
    mat[input_mat > check_size] = 0
    border = (input_mat > stroke_size) & (input_mat <= check_size)
    mat[border] = 1.0 - (input_mat[border] - stroke_size)
    return mat


def cv2pil(cv_img):
    cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGRA2RGBA)
    pil_img = Image.fromarray(cv_img.astype("uint8"))
    return pil_img


def postprocessing(
    image_path, output_path, masks, boxes, tmpdirname, white_outline=True
):
    image = cv2.imread(image_path)
    mask_array = masks.numpy()

    # Ensure the mask and image have the same dimensions
    mask_array = np.transpose(mask_array, (1, 2, 0))
    mask_array = mask_array.astype(np.uint8)
    mask_array = cv2.resize(
        mask_array, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST
    )

    if len(mask_array.shape) == 2:
        masks = [(mask_array > 0).astype(np.uint8)]
    else:
        masks = [
            (mask_array[:, :, i] > 0).astype(np.uint8)
            for i in range(mask_array.shape[2])
        ]

    # Combine the binary masks into a single mask
    mask = np.any(masks, axis=0).astype(np.uint8)

    # Create a masked image
    masked_image = cv2.bitwise_and(image, image, mask=mask)

    # Create a transparent image with an alpha channel
    output_segmented_path = f"{tmpdirname}/output_segmented.png"
    output_image = np.concatenate((masked_image, mask[:, :, np.newaxis] * 255), axis=2)
    cv2.imwrite(output_segmented_path, output_image)

    # Example usage
    bounding_boxes = boxes
    combined_image = crop_image_with_bounding_boxes(
        output_segmented_path, bounding_boxes
    )

    # Save the combined image
    combined_image_path = f"{tmpdirname}/combined_image.png"
    combined_image.save(combined_image_path, "PNG")

    img = cv2.imread(combined_image_path, cv2.IMREAD_UNCHANGED)
    if white_outline:
        img = add_outline(img, threshold=0, stroke_size=6, colors=((255, 255, 255),))
    else:
        img = add_outline(img, threshold=0, stroke_size=6, colors=((255, 0, 0),))
    scaled_img = rescale_img(np.array(img))
    write_img(scaled_img, output_path, alpha=True)

    print("chatgpt")  # TODO: dev


def crop_image_with_bounding_boxes(original_image_path, bounding_boxes):
    # Load the original image
    image = Image.open(original_image_path).convert("RGBA")

    # Find the maximum coordinates of the bounding boxes
    x_min = min(box[0] for box in bounding_boxes)
    y_min = min(box[1] for box in bounding_boxes)
    x_max = max(box[2] for box in bounding_boxes)
    y_max = max(box[3] for box in bounding_boxes)

    # Calculate the dimensions of the combined image
    combined_width = int(x_max - x_min)
    combined_height = int(y_max - y_min)

    # Create a new image to combine all bounding boxes
    combined_image = Image.new("RGBA", (combined_width, combined_height), (0, 0, 0, 0))

    # Iterate through bounding boxes and paste cropped images onto the combined image
    for box in bounding_boxes:
        x, y, x_max, y_max = box
        x = int(x)
        y = int(y)
        x_max = int(x_max)
        y_max = int(y_max)
        cropped_image = image.crop((x, y, x_max, y_max))
        combined_image.paste(cropped_image, (int(x - x_min), int(y - y_min)))

    return combined_image
