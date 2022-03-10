import pathlib

from aws_cdk import aws_lambda, core


class Notification(core.Construct):
    @property
    def handler(self) -> core.Resource:
        return self._handler

    def __init__(self, scope: core.Construct, component_id: str, **kwargs):
        super().__init__(scope, component_id, **kwargs)

        self.stack_name = self.node.try_get_context("prefix")
        self.component_id = component_id

        self._handler = self.create_lambda_function()

    def create_lambda_function(self) -> core.Resource:
        """
        Lambda Function to say hello.
        Ref: https://github.com/aws-samples/aws-cdk-examples/tree/master/python/lambda-cron
        """

        lambdaFn_id = f"{self.stack_name}-{self.component_id}-" + "lambda_handler"
        lambdaFn_path = str(pathlib.Path(__file__).resolve().parent) + "/lambdafn/sample_function/"

        lambdaFn = aws_lambda.Function(
            scope=self,
            id=lambdaFn_id,
            function_name=lambdaFn_id,
            code=aws_lambda.AssetCode(path=lambdaFn_path),
            handler="lambda_handler.lambda_handler",
            timeout=core.Duration.seconds(300),
            runtime=aws_lambda.Runtime.PYTHON_3_7,
            description="write some description for this lambda",
        )

        return lambdaFn
