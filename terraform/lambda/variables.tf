variable "BOT_API_TOKEN" {
  type = string
}

variable "NUMERO_GAGNANT" {
  type = string
}

variable "queue_name" {
  type = string
}

variable "image_tag_bot" {
  type = string
}

#variable "image_tag_bot_down" {
#  type = string
#}

variable "image_tag_bot_producer" {
  type = string
}

variable "remote_state_ecr_bucket" {
  type = string
}

variable "remote_state_ecr_key" {
  type = string
}

variable "remote_state_ecr_region" {
  type = string
}

variable "remote_state_dynamodb_bucket" {
  type = string
}

variable "remote_state_dynamodb_key" {
  type = string
}

variable "remote_state_dynamodb_region" {
  type = string
}

variable "region" {
  type = string
}
