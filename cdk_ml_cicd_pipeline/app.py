# from aws_cdk import core as cdf
import aws_cdk
from aws_cdk import App

from cdk_ml_cicd_pipeline.cdk_ml_cicd_pipeline_stack import \
    CdkMlCicdPipelineStack

app = App()

resource_prefix = app.node.try_get_context("prefix")
stack_name = resource_prefix + "-MLOps"
account = app.node.try_get_context("account")
region = app.node.try_get_context("region")

env_us_east_1 = aws_cdk.Environment(account=account, region=region)
CdkMlCicdPipelineStack(app, stack_name, env=env_us_east_1)


app.synth()
