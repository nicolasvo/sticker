resource "aws_sqs_queue" "queue" {
  name                       = var.queue_name
  max_message_size           = 2048
  message_retention_seconds  = 300
  visibility_timeout_seconds = 600
}

resource "aws_sqs_queue_policy" "queue" {
  queue_url = aws_sqs_queue.queue.id

  policy = <<POLICY
{
  "Version": "2012-10-17",
  "Id": "test",
  "Statement": [
    {
      "Sid": "test",
      "Effect": "Allow",
      "Principal": {
        "AWS": "${aws_iam_role.lambda.arn}"
      },
      "Action": "sqs:*",
      "Resource": "${aws_sqs_queue.queue.arn}"
    }
  ]
}
POLICY
}