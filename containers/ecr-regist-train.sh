# ACCOUNT_ID=`aws sts get-caller-identity --query 'Account' --output text`
ACCOUNT_ID='<your account id>' 
REGION='<your region>'
REGISTRY_URL="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com" 
ECR_REPOGITORY='sm-tf-nightly-gp'
IMAGE_TAG=':latest'
IMAGE_URI="${REGISTRY_URL}/${ECR_REPOGITORY}"

aws ecr get-login-password | docker login --username AWS --password-stdin $REGISTRY_URL
aws ecr create-repository --repository-name $ECR_REPOGITORY

docker build -t $ECR_REPOGITORY train/
docker tag "${ECR_REPOGITORY}${IMAGE_TAG}" $IMAGE_URI
docker push $IMAGE_URI

echo "Container registered. URI:${IMAGE_URI}"