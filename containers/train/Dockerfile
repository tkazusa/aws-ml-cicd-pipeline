FROM tensorflow/tensorflow:nightly-gpu

RUN pip install sagemaker-containers

# Copies the training code inside the container
# Directory structure in SageMaker training instance is followings.
# opt/
# ├ /ml
#    ├ code
#      ├ tf_model.py
#      └ train.py
# └ ml/
#  ├ input/
#    └ data/
#  └ model/
COPY tf_codes /opt/ml/code/

# SageMaker SDK で学習のために使うスクリプト(エントリーポイントと呼ばれている)を指定。
# SAGEMAKER_PROGRAM はデフォルトでは /opt/ml/code/ 以下を指す。
# 絶対パスを指定しても良い。
ENV SAGEMAKER_PROGRAM train.py