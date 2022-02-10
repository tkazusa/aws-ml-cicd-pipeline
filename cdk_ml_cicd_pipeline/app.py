from aws_cdk import core as cdk

from cdk_ml_cicd_pipeline.cdk_ml_cicd_pipeline_stack import CdkMlCicdPipelineStack

app = cdk.App()

resource_prefix = app.node.try_get_context("prefix")
stack_name = resource_prefix + "-MLOps"

CdkMlCicdPipelineStack(app, stack_name)

app.synth()