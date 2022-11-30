resource "aws_cloudwatch_log_group" "bot_producer" {
  name = "/aws/lambda/${aws_lambda_function.bot_producer.function_name}"

  retention_in_days = 7
}

resource "aws_cloudwatch_log_group" "bot" {
  name = "/aws/lambda/${aws_lambda_function.bot.function_name}"

  retention_in_days = 7
}

resource "aws_cloudwatch_log_group" "bot_down" {
  name = "/aws/lambda/${aws_lambda_function.bot_down.function_name}"

  retention_in_days = 7
}

resource "aws_lambda_function" "bot_producer" {
  function_name = "sticker-producer"
  timeout       = 60
  package_type  = "Image"
  image_uri     = "${data.terraform_remote_state.ecr.outputs.repository_url_bot_producer}:${var.image_tag_bot_producer}"

  environment {
    variables = {
      BOT_API_TOKEN = var.BOT_API_TOKEN
      QUEUE_NAME    = var.queue_name
    }
  }

  role = aws_iam_role.lambda.arn
}

resource "aws_lambda_function" "bot" {
  function_name = "sticker"
  memory_size   = 6000
  timeout       = 300
  package_type  = "Image"
  image_uri     = "${data.terraform_remote_state.ecr.outputs.repository_url_bot}:${var.image_tag_bot}"

  environment {
    variables = {
      BOT_API_TOKEN  = var.BOT_API_TOKEN
      NUMERO_GAGNANT = var.NUMERO_GAGNANT
    }
  }

  role = aws_iam_role.lambda.arn
}

resource "aws_lambda_function" "bot_down" {
  function_name = "sticker-down"
  timeout       = 60
  package_type  = "Image"
  image_uri     = "${data.terraform_remote_state.ecr.outputs.repository_url_bot_down}:${var.image_tag_bot_down}"

  environment {
    variables = {
      BOT_API_TOKEN = var.BOT_API_TOKEN
    }
  }

  role = aws_iam_role.lambda.arn
}

data "aws_iam_policy_document" "lambda" {
  statement {
    actions = ["sts:AssumeRole"]

    principals {
      type        = "Service"
      identifiers = ["lambda.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "lambda" {
  name               = "sticker"
  assume_role_policy = data.aws_iam_policy_document.lambda.json
  managed_policy_arns = [
    "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole",
    "arn:aws:iam::aws:policy/AmazonSQSFullAccess",
  ]
}

resource "aws_lambda_function_url" "bot_producer" {
  function_name      = aws_lambda_function.bot_producer.function_name
  authorization_type = "NONE"
}

resource "aws_lambda_function_url" "bot_down" {
  function_name      = aws_lambda_function.bot_down.function_name
  authorization_type = "NONE"
}

resource "aws_lambda_event_source_mapping" "bot" {
  event_source_arn = aws_sqs_queue.queue.arn
  function_name    = aws_lambda_function.bot.arn
}