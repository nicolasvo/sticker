resource "aws_apigatewayv2_api" "api_gw_bot_down" {
  name          = "api_gw_bot_down"
  protocol_type = "HTTP"
}

resource "aws_apigatewayv2_stage" "api_gw_bot_down" {
  api_id = aws_apigatewayv2_api.api_gw_bot_down.id

  name        = "default"
  auto_deploy = true

  access_log_settings {
    destination_arn = aws_cloudwatch_log_group.api_gw_bot_down.arn

    format = jsonencode({
      requestId               = "$context.requestId"
      sourceIp                = "$context.identity.sourceIp"
      requestTime             = "$context.requestTime"
      protocol                = "$context.protocol"
      httpMethod              = "$context.httpMethod"
      resourcePath            = "$context.resourcePath"
      routeKey                = "$context.routeKey"
      status                  = "$context.status"
      responseLength          = "$context.responseLength"
      integrationErrorMessage = "$context.integrationErrorMessage"
      }
    )
  }
}

resource "aws_apigatewayv2_integration" "api_gw_bot_down" {
  api_id = aws_apigatewayv2_api.api_gw_bot_down.id

  integration_uri        = aws_lambda_function.bot_down.invoke_arn
  integration_type       = "AWS_PROXY"
  integration_method     = "POST"
  payload_format_version = "2.0"
}

resource "aws_apigatewayv2_route" "route_bot_down" {
  api_id = aws_apigatewayv2_api.api_gw_bot_down.id

  route_key = "ANY /sticker-bot-down"
  target    = "integrations/${aws_apigatewayv2_integration.api_gw_bot_down.id}"
}

resource "aws_cloudwatch_log_group" "api_gw_bot_down" {
  name = "/aws/api_gw/${aws_apigatewayv2_api.api_gw_bot_down.name}"

  retention_in_days = 7
}

resource "aws_lambda_permission" "api_gw_bot_down" {
  statement_id  = "AllowExecutionFromAPIGateway"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.bot_down.function_name
  principal     = "apigateway.amazonaws.com"

  source_arn = "${aws_apigatewayv2_api.api_gw_bot_down.execution_arn}/*/*"
}

resource "aws_apigatewayv2_api" "api_gw_bot" {
  name          = "api_gw_bot"
  protocol_type = "HTTP"
}

resource "aws_apigatewayv2_stage" "api_gw_bot" {
  api_id = aws_apigatewayv2_api.api_gw_bot.id

  name        = "default"
  auto_deploy = true

  access_log_settings {
    destination_arn = aws_cloudwatch_log_group.api_gw_bot.arn

    format = jsonencode({
      requestId               = "$context.requestId"
      sourceIp                = "$context.identity.sourceIp"
      requestTime             = "$context.requestTime"
      protocol                = "$context.protocol"
      httpMethod              = "$context.httpMethod"
      resourcePath            = "$context.resourcePath"
      routeKey                = "$context.routeKey"
      status                  = "$context.status"
      responseLength          = "$context.responseLength"
      integrationErrorMessage = "$context.integrationErrorMessage"
      }
    )
  }
}

resource "aws_apigatewayv2_integration" "api_gw_bot" {
  api_id = aws_apigatewayv2_api.api_gw_bot.id

  integration_uri        = aws_lambda_function.bot.invoke_arn
  integration_type       = "AWS_PROXY"
  integration_method     = "POST"
  payload_format_version = "2.0"
}

resource "aws_apigatewayv2_route" "route_bot" {
  api_id = aws_apigatewayv2_api.api_gw_bot.id

  route_key = "ANY /sticker-bot"
  target    = "integrations/${aws_apigatewayv2_integration.api_gw_bot.id}"
}

resource "aws_cloudwatch_log_group" "api_gw_bot" {
  name = "/aws/api_gw/${aws_apigatewayv2_api.api_gw_bot.name}"

  retention_in_days = 7
}

resource "aws_lambda_permission" "api_gw_bot" {
  statement_id  = "AllowExecutionFromAPIGateway"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.bot.function_name
  principal     = "apigateway.amazonaws.com"

  source_arn = "${aws_apigatewayv2_api.api_gw_bot.execution_arn}/*/*"
}
