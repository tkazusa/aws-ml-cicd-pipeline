# FROM python:3.7-slim-buster
FROM 222136011322.dkr.ecr.us-east-1.amazonaws.com/mlops-demo-evaluate:base

COPY requirements.txt ./
RUN apt-get update && apt-get install -y build-essential
RUN pip3 install --upgrade pip
RUN pip uninstall torch torchvision -y && pip install --no-cache-dir -U -r requirements.txt
# RUN pip3 install --no-cache-dir -qU boto3 pandas==0.25.3 scikit-learn==0.20.0 torch==1.9.1 torchvision==0.10.1 pycocotools==2.0.2
COPY src/* /opt/ml/processing/input/code/

ENTRYPOINT ["python3", "/opt/ml/processing/input/code/evaluate.py"]
