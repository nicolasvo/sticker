resource "aws_cloudwatch_log_group" "bot" {
  name = "/aws/lambda/${aws_lambda_function.bot.function_name}"

  retention_in_days = 7
}

resource "aws_lambda_function" "bot" {
  function_name = "sticker-sam"
  memory_size   = 8000
  timeout       = 300
  package_type  = "Image"
  image_uri     = "${data.terraform_remote_state.ecr.outputs.repository_url_bot}:${var.image_tag_bot}"

  environment {
    variables = {
      BOT_API_TOKEN   = var.BOT_API_TOKEN
      NUMERO_GAGNANT  = var.NUMERO_GAGNANT
      SEGMENT_SAM_URL = aws_lambda_function_url.segment_sam.function_url
    }
  }

  role = aws_iam_role.lambda.arn
}

resource "aws_cloudwatch_log_group" "segment_sam" {
  name = "/aws/lambda/${aws_lambda_function.segment_sam.function_name}"

  retention_in_days = 7
}

resource "aws_lambda_function" "segment_sam" {
  function_name = "segment-sam"
  memory_size   = 8000
  timeout       = 600
  package_type  = "Image"
  image_uri     = "${data.terraform_remote_state.ecr.outputs.repository_url_segment_sam}:${var.image_tag_segment_sam}"

  environment {
    variables = {
      HOME = "/tmp"
    }
  }

  role = aws_iam_role.segment.arn
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

data "aws_iam_policy_document" "dynamodb" {
  statement {
    sid = "dynamodball"
    actions = [
      "dynamodb:List*",
      "dynamodb:DescribeReservedCapacity*",
      "dynamodb:DescribeLimits",
      "dynamodb:DescribeTimeToLive"
    ]

    resources = [
      "*",
    ]
  }
  statement {
    sid = "dynamodbtable"
    actions = [
      "dynamodb:*"
    ]

    resources = [
      data.terraform_remote_state.dynamodb.outputs.dynamodb_table_arn
    ]
  }
}

resource "aws_iam_policy" "dynamodb" {
  policy = data.aws_iam_policy_document.dynamodb.json
}

resource "aws_iam_role" "lambda" {
  name               = "sticker-sam"
  assume_role_policy = data.aws_iam_policy_document.lambda.json
  managed_policy_arns = [
    "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole",
    aws_iam_policy.dynamodb.arn
  ]
}

resource "aws_lambda_function_url" "bot" {
  function_name      = aws_lambda_function.bot.function_name
  authorization_type = "NONE"
}

resource "aws_iam_role" "segment" {
  name               = "segment"
  assume_role_policy = data.aws_iam_policy_document.lambda.json
  managed_policy_arns = [
    "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole",
  ]
}


resource "aws_lambda_function_url" "segment_sam" {
  function_name      = aws_lambda_function.segment_sam.function_name
  authorization_type = "NONE"
}
