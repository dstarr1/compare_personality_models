IMAGE_NAME='person_model'
TAG='dev'

image-cpu:
	docker build -f Dockerfile_cpu --progress plain \
		-t ${IMAGE_NAME}:${TAG} .

image-gpu:
	docker build -f Dockerfile_awsgpu --progress plain \
		-t ${IMAGE_NAME}:${TAG} .


shell-cpu: image-cpu
	docker run -it -w=/local -v $(PWD)/:/local \
		-e AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID} \
		-e AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY} \
		${IMAGE_NAME}:${TAG} bash

shell-gpu: image-gpu
	nvidia-docker run -it -w=/local -v $(PWD)/:/local \
		-e AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID} \
		-e AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY} \
		${IMAGE_NAME}:${TAG} bash
