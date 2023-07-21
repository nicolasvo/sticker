import base64
import json
import logging

from segment import segment

logger = logging.getLogger()
logger.setLevel(logging.INFO)
image_path = "/var/task/input.jpeg"
output_path = "/var/task/output.png"


def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read())
        return encoded_image.decode("utf-8")


def base64_to_image(base64_string, output_file_path):
    with open(output_file_path, "wb") as image_file:
        decoded_image = base64.b64decode(base64_string)
        image_file.write(decoded_image)


def lambda_handler(event, context):
    try:
        image = event["image"]
        text_prompt = event["text_prompt"]
        base64_to_image(image, image_path)
        masks, boxes, phrases, logits = segment(image_path, text_prompt)
        image = image_to_base64(output_path)

    except Exception as e:
        print(e)
        raise e

    return json.dumps(
        {
            "image": image,
            "masks": masks,
            "boxes": boxes,
            "phrases": phrases,
            "logits": logits,
        }
    )
