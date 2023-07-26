resource "aws_s3_bucket" "b" {
  bucket = "segment-weights"

  tags = {
    Name = "segment-weights"
  }
}

# resource "aws_s3_bucket_acl" "acl" {
#   bucket = aws_s3_bucket.b.id
#   acl    = "private"
# }
