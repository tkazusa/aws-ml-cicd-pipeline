# Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this
# software and associated documentation files (the "Software"), to deal in the Software
# without restriction, including without limitation the rights to use, copy, modify,
# merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
# PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
# HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import os
import argparse

import sys

import numpy as np
import tensorflow as tf

from tf_model import tf_model


import subprocess
import json

DEFAULT_ATTRIBUTES = (
    'index',
    'uuid',
    'name',
    'timestamp',
    'memory.total',
    'memory.free',
    'memory.used',
    'utilization.gpu',
    'utilization.memory'
)


def data_load(train_dir):
    X_train = np.load(os.path.join(train_dir, 'train.npz'))['image']
    y_train = np.load(os.path.join(train_dir, 'train.npz'))['label']
    X_test = np.load(os.path.join(train_dir, 'test.npz'))['image']
    y_test = np.load(os.path.join(train_dir, 'test.npz'))['label']
    return X_train, y_train, X_test, y_test
    

def main(args):
    train_dir = args.train
    model_dir = args.model_dir
    
    X_train, y_train, X_test, y_test = data_load(train_dir)    
    
    model = tf_model()

    callbacks = []    
    callbacks.append(
        tf.keras.callbacks.ModelCheckpoint(
            model_dir + '/checkpoint-{epoch}.h5')
    )

    model.fit(X_train, y_train,
              epochs=args.epochs,
              batch_size=args.batch_size,
              callbacks=callbacks)
    
    model.evaluate(X_test, y_test)
    
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # SageMaker SDK の Estimator で定義されたハイパーパラメータ
    parser.add_argument('--batch_size',
                        type=int,
                        default=128)
    parser.add_argument('--epochs',
                        type=int,
                        default=12)
    
    
    # SageMaker 学習用インスタンスが S3 とデータなどを共有する際のパスを環境変数から受け取っている
    # SM_MODEL_DIR /opt/ml/model
    parser.add_argument('--model_dir',
                        type=str,
                        default=os.environ['SM_MODEL_DIR'])
    
    # SM_CHANNEL_TRAIN /opt/ml/input/data/train
    parser.add_argument('--train',
                        type=str,
                        default=os.environ['SM_CHANNEL_TRAIN'])

    
    args = parser.parse_args()
    main(args)