import json
import logging

from PIL import Image


logger = logging.getLogger()
logger.setLevel(logging.INFO)


def image_to_base64(image_path):
    import base64

    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read())
        return encoded_image.decode("utf-8")


def base64_to_image(base64_string, output_file_path):
    import base64

    with open(output_file_path, "wb") as image_file:
        decoded_image = base64.b64decode(base64_string)
        image_file.write(decoded_image)


def lambda_handler(event, context):
    try:
        image = event["image"]
        output_path = "/var/task/input.jpeg"
        base64_to_image(image, output_path)

    except Exception as e:
        print(e)
        raise e

    return {
        "statusCode": 200,
        "body": json.dumps(
            {
                "message": "hello world",
            }
        ),
    }
