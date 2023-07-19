import boto3

dynamodb = boto3.resource("dynamodb")
table_name = "sticker-sam"  # TODO: dev variabilize with env var

def get_item(user_id):
    table = dynamodb.Table(table_name)
    response = table.get_item(Key={"UserId": str(user_id)})
    if response.get("Item"):
        item = response["Item"]
    else:
        item = None
    return item

def upsert_item(user_id, message_id=None, file_id=None, segmented_photo=None):
    table = dynamodb.Table(table_name)
    # Upsert an item in the table
    key = {
        "UserId": str(user_id),
    }
    update_expression = (
        "SET #fid = :file_id_val, #sp = :segmented_photo_val, #mid = :message_id_val"
    )
    expression_attribute_names = {
        "#mid": "MessageId",
        "#fid": "FileId",
        "#sp": "SegmentedPhoto",
    }
    expression_attribute_values = {
        ":message_id_val": "message123",
        ":file_id_val": "new_file_id",
        ":segmented_photo_val": "new_segmented_photo",
    }
    condition_expression = "attribute_exists(UserId)"  # Only update if UserId exists

    table.update_item(
        Key=key,
        UpdateExpression=update_expression,
        ExpressionAttributeNames=expression_attribute_names,
        ExpressionAttributeValues=expression_attribute_values,
        ConditionExpression=condition_expression,
        ReturnValues="NONE",  # You can change the ReturnValues as needed
    )

    # If the update condition fails, create a new item
    if "Attributes" not in table.update_item(
        Key=key,
        UpdateExpression="SET #mid = :message_id_val, #fid = :file_id_val, #sp = :segmented_photo_val"
        ExpressionAttributeNames=expression_attribute_names,
        ExpressionAttributeValues=expression_attribute_values,
        ConditionExpression="attribute_not_exists(UserId)",  # Only create if UserId does not exist
        ReturnValues="NONE",  # You can change the ReturnValues as needed
    ):
        item = {**key, **expression_attribute_values}
        table.put_item(
            Item=item,
            ConditionExpression="attribute_not_exists(UserId)",  # Only put the item if UserId does not exist
        )
