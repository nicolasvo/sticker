output "repository_url_bot_producer" {
  value = aws_ecr_repository.bot_producer.repository_url
}

output "repository_url_bot" {
  value = aws_ecr_repository.bot.repository_url
}
