# MLflow server instructions

* This MLFlow server is intended to be run on remote server (e.g. on AWS) since it stores data in S3.
* But it can also be run on a local desktop/laptop and accessed at http://127.0.0.1:5000/


* starting up mlflow server:
```bash
# A couple environment variables are needed by container:
export AWS_ACCESS_KEY_ID=`aws configure get aws_access_key_id`
export AWS_SECRET_ACCESS_KEY=`aws configure get aws_secret_access_key`
export MLFLOW_ARTIFACTS_URI="s3://.../..."

# Then run the mlflow server:
make run
```
