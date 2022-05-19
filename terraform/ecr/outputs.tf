output "repository_url_bot" {
  value = aws_ecr_repository.bot.repository_url
}

output "repository_url_bot_down" {
  value = aws_ecr_repository.bot_down.repository_url
}
