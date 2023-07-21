def segment_photo(image_path) -> None:
    with tempfile.TemporaryDirectory(dir="/tmp/") as tmpdirname:
        masks, boxes, _, _ = segment(image_path, update.message.caption, tmpdirname)
        postprocessing(imgae_path, masks, boxes, tmpdirname, white_outline=False)
        output_path = f"{tmpdirname}/output.png"  # TODO: dev ugly


def segment(photo_path, text_prompt, tmpdirname):
    model = LangSAM()
    image_pil = Image.open(photo_path).convert("RGB")
    output_path = f"{tmpdirname}/output_draw.png"
    masks, boxes, phrases, logits = model.predict(image_pil, text_prompt)
    labels = [f"{phrase} {logit:.2f}" for phrase, logit in zip(phrases, logits)]
    image_array = np.asarray(image_pil)
    image_draw = image = draw_image(image_array, masks, boxes, labels)
    image = Image.fromarray(np.uint8(image)).convert("RGB")
    cv2.imwrite(output_path, cv2.cvtColor(image_draw, cv2.COLOR_RGBA2BGRA))

    return masks, boxes, phrases, logits


def postprocessing(photo_path, masks, boxes, tmpdirname, white_outline=True):
    image = cv2.imread(photo_path)
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
    output_path = f"{tmpdirname}/output.png"  # TODO: dev ugly
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
