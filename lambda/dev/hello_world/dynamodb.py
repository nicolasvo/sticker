import boto3

dynamodb = boto3.resource("dynamodb")
table_name = "sticker-sam"  # TODO: dev variabilize with env var

attributes = {
    "message_id": "MessageId",
    "file_id": "FileId",
    "segmented_photo": "SegmentedPhoto",
}


def query_builder(**kwargs):
    # q, e = query_builder(message_id=123, file_id="heheh", segmented_photo=None) # TODO: dev remove
    # q, e = query_builder(message_id=123, file_id="heheh", segmented_photo="something") # TODO: dev remove
    query = "SET "
    expression = {}
    for k, v in kwargs.items():
        if v:
            query += f"{attributes[k]} = :{k}, "
            expression.update({f":{k}": v})
    query = query[:-2]
    return query, expression


def upsert_item(user_id, message_id=None, file_id=None, segmented_photo=None):
    item = get_item(user_id)
    if item:
        print("item does already exists")
        table = dynamodb.Table(table_name)
        query, expression = query_builder(
            message_id=message_id, file_id=file_id, segmented_photo=segmented_photo
        )
        response = table.update_item(
            Key={"UserId": user_id},
            UpdateExpression=query,
            ExpressionAttributeValues=expression,
            ReturnValues="UPDATED_NEW",
        )
        # response = table.update_item(
        #     Key={"UserId": user_id},
        #     UpdateExpression="SET MessageId = :message_id, FileId = :file_id, SegmentedPhoto = :segmented_photo",
        #     ExpressionAttributeValues={
        #         ":message_id": message_id,
        #         ":file_id": file_id,
        #         ":segmented_photo": segmented_photo,
        #     },
        #     ReturnValues="UPDATED_NEW",
        # )

    if not item:
        print("item does not exist yet")
        new_item = {
            "UserId": user_id,
            "MessageId": message_id,
            "FileId": file_id,
            "SegmentedPhoto": segmented_photo,
        }

        table = dynamodb.Table(table_name)
        response = table.put_item(Item=new_item)
        print(f"insert response: {response}")  # TODO: dev remove maybe


def get_item(user_id):
    table = dynamodb.Table(table_name)
    response = table.get_item(Key={"UserId": user_id})
    if response.get("Item"):
        item = response["Item"]
        print(f"user has record {item}")  # TODO: dev remove
    else:
        print("user has no record")  # TODO: dev rmeove
        item = None
    return item


def delete_item(user_id):
    table = dynamodb.Table(table_name)
    response = table.delete_item(Key={"UserId": user_id})
    print(f"delete respponse: {response}")
