from typing import Any, Dict

from aws_cdk import core as cdk

from resources.train.train import Train


class CdkMlCicdPipelineStack(cdk.Stack):
    def __init__(self, scope: cdk.Construct, stack_name: str, **kwargs: Dict[str, Any]) -> None:
        super().__init__(scope, stack_name, **kwargs)

        # ----- Notification Component ------
        train = Train(scope=self, stack_name=stack_name, component_id="train")
