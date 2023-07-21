import base64
def image_to_base64(image_path):
    import base64

    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read())
        return encoded_image.decode("utf-8")

import requests
payload = {
    "image": image_to_base64("IMG_3301.jpeg")
}


r = requests.post("http://localhost:9000/2015-03-31/functions/function/invocations", json=payload, timeout=600)
