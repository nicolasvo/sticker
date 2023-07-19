data "terraform_remote_state" "ecr" {
  backend = "s3"
  config = {
    bucket = var.remote_state_ecr_bucket
    key    = var.remote_state_ecr_key
    region = var.remote_state_ecr_region
  }
}

data "terraform_remote_state" "dynamodb" {
  backend = "s3"
  config = {
    bucket = var.remote_state_dynamodb_bucket
    key    = var.remote_state_dynamodb_key
    region = var.remote_state_dynamodb_region
  }
}
