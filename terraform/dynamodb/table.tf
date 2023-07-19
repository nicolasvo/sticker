resource "aws_dynamodb_table" "basic-dynamodb-table" {
  name           = "sticker-sam"
  billing_mode   = "PROVISIONED"
  read_capacity  = 20
  write_capacity = 20
  hash_key       = "UserId"

  attribute {
    name = "UserId"
    type = "S"
  }

  tags = {
    Name        = "sticker-sam"
    Environment = "production"
  }
}