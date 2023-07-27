import base64
import json
import logging
import os
import zlib

import boto3
import numpy as np

from segment import segment

logger = logging.getLogger()
logger.setLevel(logging.INFO)
image_path = "/tmp/input.jpeg"
output_path = "/tmp/output.png"
bucket_weights = os.getenv("BUCKET_WEIGHTS")
bucket_weights_folder = os.getenv("BUCKET_WEIGHTS_FOLDER")
initialized = False


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

            # Create a local file path within /tmp/.cache directory for the object
            local_file_path = os.path.join(
                "/tmp/.cache", os.path.relpath(object_key, s3_folder_key)
            )

            # Create the directory structure in /tmp/.cache if it doesn't exist
            os.makedirs(os.path.dirname(local_file_path), exist_ok=True)

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


def initialize_once():
    # Put your one-time initialization code here
    print("Initialization code running...")
    copy_s3_folder_to_lambda(bucket_weights, bucket_weights_folder)
    # For example, you might want to establish connections, load models, or set up configurations

    # Set the global variable to True to mark the initialization as done
    global initialized
    initialized = True


def lambda_handler(event, context):
    try:
        global initialized

        # Check if initialization has been done
        if not initialized:
            # Run the initialization code
            initialize_once()
        body = json.loads(event["body"])
        image = body["image"]
        text_prompt = body["text_prompt"]
        base64_to_image(image, image_path)
        masks, boxes, phrases, logits = segment(image_path, text_prompt)
        # masks_ = masks.tolist()
        boxes_ = boxes.tolist()
        logits_ = logits.tolist()
        image_mask_np = masks.numpy()
        compressed_np = zlib.compress(image_mask_np.tobytes())
        compressed_base64 = base64.b64encode(compressed_np).decode("utf-8")

    except Exception as e:
        print(e)
        raise e

    return json.dumps(
        {
            "masks": compressed_base64,
            "masks_shape": masks.shape,
            "boxes": boxes_,
            "phrases": phrases,
            "logits": logits_,
        }
    )
