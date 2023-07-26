import base64
import json
import logging
import os

import boto3

from segment import segment

logger = logging.getLogger()
logger.setLevel(logging.INFO)
image_path = "/tmp/input.jpeg"
output_path = "/tmp/output.png"
bucket = os.getenv("BUCKET")
bucket_folder = os.getenv("BUCKET_FOLDER")


def copy_s3_folder_to_lambda(s3_bucket, s3_folder_key):
    print("Copying weights folder")
    # Create a Boto3 client for S3
    s3_client = boto3.client("s3")

    # List all objects in the given S3 folder
    response = s3_client.list_objects_v2(Bucket=s3_bucket, Prefix=s3_folder_key)

    # Check if the folder exists and has objects
    if "Contents" in response:
        for item in response["Contents"]:
            # Get the object's key (filename)
            object_key = item["Key"]

            # Create a local file path within /tmp directory for the object
            local_file_path = os.path.join("/tmp/.cache")

            # Download the object to the local file path
            s3_client.download_file(s3_bucket, object_key, local_file_path)


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
        copy_s3_folder_to_lambda(bucket, bucket_folder)
        body = json.loads(event["body"])
        image = body["image"]
        text_prompt = body["text_prompt"]
        base64_to_image(image, image_path)
        masks, boxes, phrases, logits = segment(image_path, text_prompt)
        image = image_to_base64(output_path)
        masks_ = masks.tolist()
        boxes_ = boxes.tolist()
        logits_ = logits.tolist()

    except Exception as e:
        print(e)
        raise e

    return json.dumps(
        {
            "image": image,
            "masks": masks_,
            "boxes": boxes_,
            "phrases": phrases,
            "logits": logits_,
        }
    )
