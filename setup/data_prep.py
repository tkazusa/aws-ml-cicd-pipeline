import os
import numpy as np
from keras.datasets import mnist
import sagemaker

bucket = '<your bucket name>'
sagemaker_session = sagemaker.Session()

(x_train, y_train), (x_test, y_test) = mnist.load_data()

os.makedirs("./data", exist_ok = True)
np.savez('./data/train', image=x_train, label=y_train)
np.savez('./data/test', image=x_test, label=y_test)

input_train = sagemaker_session.upload_data('data', bucket=bucket)