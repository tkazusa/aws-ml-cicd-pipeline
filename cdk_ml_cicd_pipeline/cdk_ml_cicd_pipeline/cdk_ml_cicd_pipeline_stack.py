from typing import Any, Dict

# from aws_cdk import core as cdk
from aws_cdk import Stack
from constructs import Construct

from resources.deploy.deploy import Deploy
from resources.train.train import Train
from resources.visualization.mlflow.mlflow_tracking import MLflowTracking


class CdkMlCicdPipelineStack(Stack):
    # def __init__(self, scope: cdk.Construct, stack_name: str, **kwargs: Dict[str, Any]) -> None:
    #     super().__init__(scope, stack_name, **kwargs)
    def __init__(self, scope: Construct, stack_name: str, **kwargs: Dict[str, Any]) -> None:
        super().__init__(scope, stack_name, **kwargs)

        # ----- Training Pipeline ------
        train = Train(scope=self, stack_name=stack_name, component_id="train")

        # ----- MLflow Tracking Server on AWS Fargate ------
        mlflow_tracking = MLflowTracking(scope=self, stack_name=stack_name, component_id="mlflow")

        # ----- Deploy Component ------
        deploy = Deploy(scope=self, stack_name=stack_name, component_id="deploy")
