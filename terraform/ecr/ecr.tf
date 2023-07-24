resource "aws_ecr_repository" "bot" {
  name                 = "sticker-sam"
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }
}

resource "aws_ecr_repository" "segment_sam" {
  name                 = "segment-sam"
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }
}

