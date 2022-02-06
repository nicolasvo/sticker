resource "aws_lambda_function" "bot" {
  function_name = "sticker-bot"
  memory_size   = 2048
  timeout       = 300
  package_type  = "Image"
  image_uri     = "${data.terraform_remote_state.ecr.outputs.repository_url_bot}:${var.image_tag_bot}"

  environment {
    variables = {
      BOT_API_TOKEN  = var.BOT_API_TOKEN
      NUMERO_GAGNANT = var.NUMERO_GAGNANT
    }
  }

  role = aws_iam_role.lambda_exec_bot.arn
}

resource "aws_cloudwatch_log_group" "bot" {
  name = "/aws/lambda/${aws_lambda_function.bot.function_name}"

  retention_in_days = 7
}

resource "aws_iam_role" "lambda_exec_bot" {
  name = "lambda_bot"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Sid    = ""
      Principal = {
        Service = "lambda.amazonaws.com"
      }
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "lambda_policy_bot" {
  role       = aws_iam_role.lambda_exec_bot.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

resource "aws_lambda_function" "bot_down" {
  function_name = "sticker-bot-down"
  timeout       = 60
  package_type  = "Image"
  image_uri     = "${data.terraform_remote_state.ecr.outputs.repository_url_bot_down}:${var.image_tag_bot_down}"

  environment {
    variables = {
      BOT_API_TOKEN = var.BOT_API_TOKEN
    }
  }

  role = aws_iam_role.lambda_exec_bot_down.arn
}

resource "aws_cloudwatch_log_group" "bot_down" {
  name = "/aws/lambda/${aws_lambda_function.bot_down.function_name}"

  retention_in_days = 7
}

resource "aws_iam_role" "lambda_exec_bot_down" {
  name = "lambda_bot_down"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Sid    = ""
      Principal = {
        Service = "lambda.amazonaws.com"
      }
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "lambda_policy_bot_down" {
  role       = aws_iam_role.lambda_exec_bot_down.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}
