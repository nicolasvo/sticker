# aws documentation

https://aws.amazon.com/blogs/aws/new-for-aws-lambda-container-image-support/

# steps

## set variables

terraform/terraform.tfvars:
- BOT_API_TOKEN
- NUMERO_GAGNANT
- image_tag_bot
- image_tag_bot_down
- remote_state_ecr_bucket
- remote_state_ecr_key 
- remote_state_ecr_region
- provider_region

terraform/lambda/backend.tfvars:
- bucket
- key
- region

terraform/ecr/backend.tfvars:
- bucket
- key
- region

terraform init -backend-config=backend.tfvars

## aws login

- aws login to create config
- set AWS_PROFILE

## docker login

aws ecr get-login-password --region region | docker login --username AWS --password-stdin repository_url
docker build -t repository_url:tag .
docker push image:tag

## deployment

- docker: build image
- terraform: apply ecr
- docker: push image
- terraform: apply lambda
- set bot webhook

# dev setup

1. build image and run container with 9000:8080 port forwarding: `docker run -it --rm -p 9000:8080 test-lambda:latest`
2. `ngrok http 9000`
3. set bot webhook
4. write to bot

# TODO
- case where no segmentation was made
- increase ram to da max

# pascal classes

person
car
aeroplane
bicycle
bird
boat
bottle
bus
cat
chair
cow
diningtable
dog = car
horse
motorbike
pottedplant
sheep
sofa
train
monitor
