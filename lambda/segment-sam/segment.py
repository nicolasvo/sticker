import os

import boto3
import cv2
import numpy as np
from PIL import Image

from lang_sam import LangSAM
from lang_sam.utils import draw_image

output_path = "/tmp/output.png"
image_path = "/tmp/input.jpeg"
bucket_images = os.getenv("BUCKET_IMAGES")


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


def segment(image_path, text_prompt):
    # TODO: if no text prompt, segment everything
    masks, boxes, phrases, logits = _segment(image_path, text_prompt)
    return masks, boxes, phrases, logits


def _segment(photo_path, text_prompt):
    model = LangSAM()
    image_pil = Image.open(photo_path).convert("RGB")
    masks, boxes, phrases, logits = model.predict(image_pil, text_prompt)
    labels = [f"{phrase} {logit:.2f}" for phrase, logit in zip(phrases, logits)]
    image_array = np.asarray(image_pil)
    image_draw = image = draw_image(image_array, masks, boxes, labels)
    image = Image.fromarray(np.uint8(image)).convert("RGB")
    # cv2.imwrite(output_path, cv2.cvtColor(image_draw, cv2.COLOR_RGBA2BGRA))
    # image_url = upload_file_and_get_presigned_url(bucket_images, f"output.png", output_path)

    return masks, boxes, phrases, logits
