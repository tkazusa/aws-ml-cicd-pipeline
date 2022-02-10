
FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:1.8.0-gpu-py36
 
 COPY requirements.txt ./
# RUN pip install  numpy
RUN pip uninstall torch torchvision -y && pip install --no-cache-dir -U -r requirements.txt

# WORKDIR /
COPY src /opt/ml/code/
ENV SAGEMAKER_PROGRAM train.py