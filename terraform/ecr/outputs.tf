output "repository_url_bot" {
  value = aws_ecr_repository.bot.repository_url
}

output "repository_url_segment_sam" {
  value = aws_ecr_repository.segment_sam.repository_url
}
