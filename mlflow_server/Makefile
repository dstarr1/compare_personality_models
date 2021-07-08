IMAGE_NAME='mlflow_server'

image:
	docker build --progress plain -f Dockerfile_mlflowserver \
		--build-arg MLFLOW_ARTIFACTS_URI=${MLFLOW_ARTIFACTS_URI} \
		-t ${IMAGE_NAME} .

shell: image
	docker run -it -p 5000:5000 -v $(PWD)/shared_vol:/shared_vol \
		-e AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID} \
		-e AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY} \
		${IMAGE_NAME} bash

run: image
	docker run -p 5000:5000 -v $(PWD)/shared_vol:/shared_vol \
		-e AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID} \
		-e AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY} \
		${IMAGE_NAME}
