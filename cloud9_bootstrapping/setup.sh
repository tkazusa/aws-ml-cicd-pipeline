#!/bin/bash
# Simple bash for Demoset C9 Deployment. 
# NOT FOR PRODUCTION USE - Only for Demoset purposes
C9STACK="DemosetC9"
CFNFILE="cloud9-cfn.yaml"
PROFILE=$2
REGION=$1 

if ! [ -x "$(command -v aws)" ]; then
  echo 'Error: aws cli is not installed.' >&2
  exit 1
fi

if [ ! -z  "$PROFILE" ]; then
  export AWS_PROFILE=$PROFILE
fi

if [ ! -z "$REGION" ]; then
  export AWS_DEFAULT_REGION=$REGION
fi


is_exist=$(aws cloudformation describe-stacks --stack-name $C9STACK >& /dev/null)
if [ $? -eq 0 ]; then
  echo "[UPDATE] Stack $C9STACK is already exist."
  aws cloudformation update-stack --stack-name $C9STACK --template-body file://$CFNFILE --capabilities CAPABILITY_NAMED_IAM
  aws cloudformation wait stack-update-complete --stack-name $C9STACK
  echo "Cloud9 Instance is Updated!!!"
else
  echo "[CREATE] Stack $C9STACK is not exist."
  aws cloudformation create-stack --stack-name $C9STACK --template-body file://$CFNFILE --capabilities CAPABILITY_NAMED_IAM
  aws cloudformation wait stack-create-complete --stack-name $C9STACK
  echo "Cloud9 Instance is Ready!!!"
fi

exit 0 