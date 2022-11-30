ecr=

push:
	docker buildx build --platform linux/amd64 -t ${ecr}:$(tag) .
	docker push ${ecr}:$(tag)
	cd ../../terraform/lambda/; terraform apply -auto-approve

