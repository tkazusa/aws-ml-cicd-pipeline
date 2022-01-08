import logging
import os
import time

import boto3
import sagemaker
import stepfunctions
import yaml
from sagemaker import get_execution_role
from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
from sagemaker.processing import ProcessingInput, ProcessingOutput, Processor
from stepfunctions.inputs import ExecutionInput
from stepfunctions.steps import Chain, ProcessingStep, TrainingStep
from stepfunctions.workflow import Workflow

stepfunctions.set_stream_logger(level=logging.INFO)
config_name = "flow.yaml"


def get_parameters():
    params = {}
    with open(config_name) as file:
        config = yaml.safe_load(file)
        params["sagemaker-role"] = config["config"]["sagemaker-role"]
        params["sfn-workflow-name"] = config["config"]["sfn-workflow-name"]
        params["sfn-role-arn"] = config["config"]["sfn-role-arn"]
        params["job-name-prefix"] = config["config"]["job-name-prefix"]
        params["prep-job-name"] = os.environ["PREP_JOB_NAME"]
        params["prep-image-uri"] = os.environ["PREPRO_IMAGE_URI"]
        params["prep-input-path"] = config["preprocess"]["input-data-path"]
        params["prep-output-path"] = config["preprocess"]["output-data-path"]
        params["train-job-name"] = os.environ["TRAIN_JOB_NAME"]
        params["train-image-uri"] = os.environ["TRAIN_IMAGE_URI"]
        params["train-output-path"] = config["train"]["output-path"]
        params["hyperparameters"] = {}
        params["hyperparameters"]["batch-size"] = config["train"]["hyperparameters"]["batch-size"]
        params["hyperparameters"]["epoch"] = config["train"]["hyperparameters"]["epoch"]
        params["eval-job-name"] = os.environ["EVAL_JOB_NAME"]
        params["eval-image-uri"] = os.environ["EVALUATE_IMAGE_URI"]
        params["eval-data-path"] = config["evaluate"]["data-path"]
        params["eval-result-path"] = config["evaluate"]["result-path"]

        # !!!!!
        # params['prep-image-uri'] = '420964472730.dkr.ecr.ap-northeast-1.amazonaws.com/mlops-demo-prepro:e6d3acaf876c63271f7b7c5101c8ea5a399acd1e'
        # params['train-image-uri'] = '420964472730.dkr.ecr.ap-northeast-1.amazonaws.com/mlops-demo-train:e6d3acaf876c63271f7b7c5101c8ea5a399acd1e'
        # params['eval-image-uri'] = '420964472730.dkr.ecr.ap-northeast-1.amazonaws.com/mlops-demo-evaluate:e6d3acaf876c63271f7b7c5101c8ea5a399acd1e'

        print("------------------")
        print(params)
    return params


def create_prepro_processing(params, job_name, sagemaker_role):
    prepro_repository_uri = params["prep-image-uri"]

    pre_processor = Processor(
        role=sagemaker_role,
        image_uri=prepro_repository_uri,
        instance_count=1,
        instance_type="ml.m5.xlarge",
        volume_size_in_gb=16,
        volume_kms_key=None,
        output_kms_key=None,
        max_runtime_in_seconds=86400,  # default is 24 hours(60*60*24)
        sagemaker_session=None,
        env=None,
        tags=None,
        network_config=None,
    )
    return pre_processor


def create_prepro_step(params, pre_processor, execution_input):
    prepro_input_data = params["prep-input-path"]
    prepro_output_data = params["prep-output-path"]
    input_dir = "/opt/ml/processing/input"
    output_dir = "/opt/ml/processing/output"

    prepro_inputs = [ProcessingInput(source=prepro_input_data, destination=input_dir, input_name="input-data")]

    prepro_outputs = [
        ProcessingOutput(
            source=output_dir,
            destination=prepro_output_data,
            output_name="processed-data",
        )
    ]

    processing_step = ProcessingStep(
        "SageMaker pre-processing step",
        processor=pre_processor,
        job_name=execution_input["PreprocessingJobName"],
        inputs=prepro_inputs,
        outputs=prepro_outputs,
        container_arguments=["--input-dir", input_dir, "--output-dir", output_dir],
    )
    return processing_step


def create_estimator(params, sagemaker_role):
    train_repository_uri = params["train-image-uri"]
    instance_type = "ml.p3.2xlarge"

    metric_definitions = [
        {
            "Name": "val:mAP",
            "Regex": "Average Precision  \(AP\) \@\[ IoU=0.50:0.95 \| area=   all \| maxDets=100 \] = ([0-9\\.]+)",
        }
    ]
    estimator = Estimator(
        image_uri=train_repository_uri,
        role=sagemaker_role,
        metric_definitions=metric_definitions,
        instance_count=1,
        instance_type=instance_type,
        hyperparameters={
            "batch-size": params["hyperparameters"]["batch-size"],
            "test-batch-size": 4,
            "lr": 0.01,
            "epochs": params["hyperparameters"]["epoch"],
        },
        output_path=params["train-output-path"],
    )

    return estimator


def create_training_step(params, estimator, execution_input):
    prepro_output_data = params["prep-output-path"]
    training_input = TrainingInput(s3_data=prepro_output_data, input_mode="FastFile")

    training_step = TrainingStep(
        "SageMaker Training Step",
        estimator=estimator,
        data={"training": training_input},
        job_name=execution_input["TrainingJobName"],
        wait_for_completion=True,
    )

    return training_step


def create_evaluation_processor(params, sagemaker_role):
    evaluation_repository_uri = params["eval-image-uri"]
    model_evaluation_processor = Processor(
        image_uri=evaluation_repository_uri,
        role=sagemaker_role,
        instance_count=1,
        instance_type="ml.p3.2xlarge",
        max_runtime_in_seconds=1200,
    )
    return model_evaluation_processor


def create_evaluation_step(params, model_evaluation_processor, execution_input, job_name, train_job_name):
    evaluation_output_destination = os.path.join(params["eval-result-path"], job_name)
    prepro_input_data = params["prep-input-path"]
    trained_model_data = os.path.join(params["train-output-path"], train_job_name, "output/model.tar.gz")
    model_dir = "/opt/ml/processing/model"
    data_dir = "/opt/ml/processing/test"
    output_dir = "/opt/ml/processing/evaluation"

    inputs_evaluation = [
        # data path for model evaluation
        ProcessingInput(
            source=prepro_input_data,
            destination=data_dir,
            input_name="data-dir",
        ),
        # model path
        ProcessingInput(
            source=trained_model_data,
            destination=model_dir,
            input_name="model-dir",
        ),
    ]

    outputs_evaluation = [
        ProcessingOutput(
            source=output_dir,
            destination=evaluation_output_destination,
            output_name="output-dir",
        ),
    ]

    evaluation_step = ProcessingStep(
        "SageMaker Evaluation step",
        processor=model_evaluation_processor,
        job_name=execution_input["EvaluationJobName"],
        inputs=inputs_evaluation,
        outputs=outputs_evaluation,
        container_arguments=["--data-dir", data_dir, "--model-dir", model_dir, "--output-dir", output_dir],
    )

    return evaluation_step


def create_sfn_workflow(params, steps):
    sfn_workflow_name = params["sfn-workflow-name"]
    workflow_execution_role = params["sfn-role-arn"]

    workflow_graph = Chain(steps)

    branching_workflow = Workflow(
        name=sfn_workflow_name,
        definition=workflow_graph,
        role=workflow_execution_role,
    )

    branching_workflow.create()
    branching_workflow.update(workflow_graph)

    time.sleep(5)

    return branching_workflow


if __name__ == "__main__":
    params = get_parameters()

    # 暫定的にプロセスIDの代わりにタイムスタンプを使用
    # from datetime import datetime
    # from dateutil import tz

    # JST = tz.gettz('Asia/Tokyo')

    # timestamp = datetime.now(tz=JST).strftime('%Y%m%d-%H%M%S')

    job_name_prefix = params["job-name-prefix"]
    # job_name = job_name_prefix + '-' + timestamp

    sagemaker_role = params["sagemaker-role"]
    # prepro_job_name = 'prepro-' + job_name
    # train_job_name = 'train-' + job_name
    # eval_job_name = 'eval-' + job_name
    prepro_job_name = params["prep-job-name"]
    train_job_name = params["train-job-name"]
    eval_job_name = params["eval-job-name"]

    execution_input = ExecutionInput(
        schema={
            "PreprocessingJobName": str,
            "TrainingJobName": str,
            "EvaluationJobName": str,
        }
    )

    pre_processor = create_prepro_processing(params, prepro_job_name, sagemaker_role)
    processing_step = create_prepro_step(params, pre_processor, execution_input)

    estimator = create_estimator(params, sagemaker_role)
    training_step = create_training_step(params, estimator, execution_input)

    model_evaluation_processor = create_evaluation_processor(params, sagemaker_role)
    evaluation_step = create_evaluation_step(
        params, model_evaluation_processor, execution_input, eval_job_name, train_job_name
    )

    branching_workflow = create_sfn_workflow(params, [processing_step, training_step, evaluation_step])

    # Execute workflow
    execution = branching_workflow.execute(
        inputs={
            # Each pre processing job requires a unique name
            "PreprocessingJobName": prepro_job_name,
            "TrainingJobName": train_job_name,
            "EvaluationJobName": eval_job_name,
        }
    )

#     # SFn の実行に必要な情報を渡す際のスキーマを定義します
#     schema = {'TrainJobName': str}
#     execution_input = ExecutionInput(schema=schema)

#     # SFn のワークフローの定義を記載します
#     inputs = {'TrainJobName': TRAINING_JOB_NAME}

#     # SageMaker の学習ジョブを実行するステップ
#     estimator = create_estimator()
#     data_path = {'train': args.data_path}

#     training_step = steps.TrainingStep(
#         'Train Step',
#         estimator=estimator,
#         data=data_path,
#         job_name=execution_input['TrainJobName'],
#         wait_for_completion=True
#     )

#     # 各 Step を連結
#     chain_list = [training_step]
#     workflow_definition = steps.Chain(chain_list)

#     # Workflow の作成
#     workflow = Workflow(
#         name=FLOW_NAME,
#         definition=workflow_definition,
#         role=WORKFLOW_ROLE,
#         execution_input=execution_input
#     )
#     workflow.create()

#     # Workflow の実行
#     execution = workflow.execute(inputs=inputs)
