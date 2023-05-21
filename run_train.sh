export FILE="resnet152_8xb32" && \ 
AWS_ACCESS_KEY_ID="" \
AWS_SECRET_ACCESS_KEY="" \
AWS_DEFAULT_REGION="eu-central-1" \
MLFLOW_TRACKING_URI="" \
MLFLOW_S3_ENDPOINT_URL="http://s3.eu-central-1.amazonaws.com" \
MLFLOW_TRACKING_USERNAME="" \
MLFLOW_TRACKING_PASSWORD="" \
python hyper_opt.py configs/$FILE.py