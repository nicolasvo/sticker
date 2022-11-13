import os
import json
import boto3


def handler(event, context):
    sqs = boto3.resource("sqs")
    queue = sqs.get_queue_by_name(
        QueueName=os.getenv("QUEUE_NAME"),
    )
    update = json.dumps(event["body"])
    response = queue.send_message(MessageBody=update)
    print(response)

    return 200
