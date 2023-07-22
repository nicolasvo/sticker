output "url_bot" {
  value = aws_lambda_function_url.bot.function_url
}

output "url_segment_sam" {
  value = aws_lambda_function_url.segment_sam.function_url
}
