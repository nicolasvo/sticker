output "url_bot_producer" {
  value = aws_lambda_function_url.bot_producer.function_url
}

output "url_bot_down" {
  value = aws_lambda_function_url.bot_down.function_url
}
