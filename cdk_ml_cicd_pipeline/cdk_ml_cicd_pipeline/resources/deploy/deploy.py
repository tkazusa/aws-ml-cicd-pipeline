import os
import pathlib

import yaml
from aws_cdk import (Aws, CfnOutput, Duration, RemovalPolicy, Resource,
                     aws_codebuild, aws_codecommit, aws_ecr, aws_iam,
                     aws_lambda, aws_sns)
from aws_cdk import aws_stepfunctions as aws_sf
from aws_cdk import aws_stepfunctions_tasks as aws_sf_tasks
from aws_cdk.aws_dynamodb import Attribute, AttributeType, Table
from constructs import Construct


class Deploy(Construct):

    def __init__(self, scope: Construct, stack_name: str, component_id: str, **kwargs):
        super().__init__(scope=scope, id=component_id, **kwargs)

        self.stack_name = stack_name
        self.component_id = component_id
        # Greengrassコンポーネントの名前。Recipeと合わせておく必要があります
        self.greengrass_component_name = "com.example.ggmlcomponent"

        self.create_deploy_pipeline()

        CfnOutput(self, 
            id="ComponentCodeRepositoryURI",
            export_name="ComponentCodeRepositoryURI",
            value=self._component_source_repository.repository_clone_url_grc)

        CfnOutput(self, 
            id="ComponentBaseImageRepositoryURI",
            export_name="ComponentBaseImageRepositoryURI",
            value=self._component_base_ecr.repository_uri)

    def get_role_name(self, name: str):
        return f"{self.stack_name}_{self.component_id}_{name}_role"

    def get_lambda_name(self, name: str):
        return f"{self.stack_name}_{self.component_id}_{name}"

    def get_lambda_path(self, name: str):
        return str(pathlib.Path(__file__).resolve().parent) + "/lambdafn/" + name + "/"

    def create_deploy_status_table(self) -> Resource:
        """コンポーネントの作成状況を保持するテーブル

        Table Schema
        * Partition_key
            * component_name
        * Sort_key
            * version
        * Item
            * bucket
            * s3_path
            * component_arn
            * pipeline_status
                * image_creating: コンテナイメージをビルド中
                * image_faild: コンテナイメージの作成に失敗
                * image_exists: コンテナイメージが存在する
                * component_exists: GGのコンポーネントが存在する
                * component_faild: 何らかの理由でコンポーネントの登録に失敗した
                * create_deployment: 
            * update_time
            * deployment_status 
                * IN_PROGRESS
                * ACTIVE
                * CANCELLED
            * deploy_group
            * job_id

        Returns:
            Resource: DynamoDB Table
        """
        table_name = f"{self.stack_name}_{self.component_id}_" + "deploy_status"
        table = Table(
            self,
            id=table_name,
            table_name=table_name,
            partition_key=Attribute(name="component_name", type=AttributeType.STRING),  # パーテーションキー
            sort_key=Attribute(name="version", type=AttributeType.STRING),  # ソートキー
            removal_policy=RemovalPolicy.DESTROY,  # Stackの削除と一緒にテーブルを削除する(オプション)
        )
        
        return table

    def create_component_source_repository(self) -> Resource:
        """Greengrassのコンポーネント内で利用するソースコードを管理するリポジトリ

        Returns:
            Resource: CodeCommit
        """
        name = f"{self.stack_name}_{self.component_id}_component-source".lower()
        repo = aws_codecommit.Repository(
            self,
            id=name,
            repository_name=name
        )
        return repo

    def create_ecr_component_base_repository(self) -> Resource:
        """コンポーネントを毎回0からビルドすると時間がかかるので、ベースになるイメージを作成

        Returns:
            Resource: ECR
        """
        ecr = aws_ecr.Repository(
            self,
            id=f"{self.stack_name}_{self.component_id}_component_base".lower(),
            repository_name="base." + self.greengrass_component_name,
            removal_policy=RemovalPolicy.DESTROY
        )
        return ecr

    def create_ecr_component_repository(self) -> Resource:
        """Greengrassのコンポーネント用にビルドしたdockerイメージを保存するレジストリ

        Returns:
            Resource: ECR
        """
        ecr = aws_ecr.Repository(
            self,
            id=f"{self.stack_name}_{self.component_id}_component".lower(),
            repository_name=self.greengrass_component_name,
            removal_policy=RemovalPolicy.DESTROY
        )
        return ecr

    def create_docker_image_buildproject(self) -> Resource:
        """Greengrassのコンポーネント用に推論アプリのdockerイメージをビルドするcodebuild

        Returns:
            Resource: codebuild
        """

        codebuild_name = f"{self.stack_name}_{self.component_id}_build_component"
        role_name = self.get_role_name("codebuild")
        codebuild_role = aws_iam.Role(
            self,
            id=role_name,
            assumed_by =aws_iam.ServicePrincipal("codebuild.amazonaws.com"),
            role_name=role_name,
            path="/service-role/"
        )
        codebuild_role.attach_inline_policy(
            aws_iam.Policy(self, "DefaultCodeBuildPermissions",
                document=aws_iam.PolicyDocument(
                    statements=[
                        aws_iam.PolicyStatement(
                            actions=[
                                "logs:CreateLogGroup",
                                "logs:CreateLogStream",
                                "logs:PutLogEvents"
                            ],
                            resources=[
                                f"arn:aws:logs:{Aws.REGION}:{Aws.ACCOUNT_ID}:log-group:/aws/codebuild/{codebuild_name}",
                                f"arn:aws:logs:{Aws.REGION}:{Aws.ACCOUNT_ID}:log-group:/aws/codebuild/{codebuild_name}:*"
                            ]
                        ),
                        aws_iam.PolicyStatement(
                            actions=[
                                "codebuild:CreateReportGroup",
                                "codebuild:CreateReport",
                                "codebuild:UpdateReport",
                                "codebuild:BatchPutTestCases",
                                "codebuild:BatchPutCodeCoverages"
                            ],
                            resources=[
                                f"arn:aws:codebuild:{Aws.REGION}:{Aws.ACCOUNT_ID}:report-group/{codebuild_name}-*"
                            ]
                        ),
                        aws_iam.PolicyStatement(
                            actions=[
                                "s3:PutObject",
                                "s3:GetObject",
                                "s3:GetObjectVersion",
                                "s3:GetBucketAcl",
                                "s3:GetBucketLocation"                            ],
                            resources=[
                                "arn:aws:s3:::{}/*".format("ml-model-build-input-us-east-1")
                            ]
                        ),
                        aws_iam.PolicyStatement(
                            actions=["ecr:GetAuthorizationToken"],
                            resources=["*"]
                        ),
                        aws_iam.PolicyStatement(
                            actions=[
                                "ecr:BatchCheckLayerAvailability",
                                "ecr:GetDownloadUrlForLayer",
                                "ecr:GetRepositoryPolicy",
                                "ecr:DescribeRepositories",
                                "ecr:ListImages",
                                "ecr:DescribeImages",
                                "ecr:BatchGetImage",
                                "ecr:InitiateLayerUpload",
                                "ecr:UploadLayerPart",
                                "ecr:CompleteLayerUpload",
                                "ecr:PutImage"
                            ],
                            resources=[
                                self._component_ecr.repository_arn,
                                self._component_base_ecr.repository_arn
                                ]
                        ),
                        aws_iam.PolicyStatement(
                            actions=[
                                "codecommit:GitPull"
                            ],
                            resources=[self._component_source_repository.repository_arn]
                        )
                    ]
                )
            )
        )

        buildspecfile = os.path.dirname(__file__) + "/buildspec/componentimage.yaml"
        with open(buildspecfile, "r") as yml:
            buildspec = yaml.safe_load(yml)
    
        code_build = aws_codebuild.Project(
            self,
            id=codebuild_name,
            project_name=codebuild_name,
            build_spec=aws_codebuild.BuildSpec.from_object(buildspec),
            environment=aws_codebuild.BuildEnvironment(
                privileged=True,
                build_image=aws_codebuild.LinuxBuildImage.STANDARD_4_0
            ),
            description='Greengrass用の推論アプリコンポーネントイメージを作成',
            timeout=Duration.minutes(60),
            role=codebuild_role
        )

        return code_build

    def create_lambda_build_image(self) -> Resource:
        """Greengrassのコンポーネント用に推論アプリのdockerイメージをビルドするcodebuildを実行するLambda

        Returns:
            Resource: lambda
        """

        lambdaFn_name = self.get_lambda_name("build_image")
        role_name = self.get_role_name("build_image")

        lambda_role = aws_iam.Role(
            self,
            id=role_name,
            assumed_by =aws_iam.ServicePrincipal("lambda.amazonaws.com"),
            role_name=role_name,
            path="/service-role/",
            managed_policies=[
                aws_iam.ManagedPolicy.from_aws_managed_policy_name("service-role/AWSLambdaBasicExecutionRole"),
                aws_iam.ManagedPolicy.from_aws_managed_policy_name("AWSCodeBuildDeveloperAccess")
            ]
        )
        lambda_role.attach_inline_policy(
            aws_iam.Policy(self, "AllowDynamoDBAccess",
                document=aws_iam.PolicyDocument(
                    statements=[
                        aws_iam.PolicyStatement(
                            actions=[
                                "dynamodb:PutItem",
                                "dynamodb:GetItem",
                                "dynamodb:UpdateItem"
                            ],
                            resources=[self._table.table_arn]
                        )
                    ]
                )
            )
        )
        lambdaFn_path = self.get_lambda_path("build_image")

        lambdaFn = aws_lambda.Function(
            self,
            id=lambdaFn_name,
            function_name=lambdaFn_name,
            code=aws_lambda.AssetCode(path=lambdaFn_path),
            handler="lambda_handler.handler",
            timeout=Duration.seconds(10),
            runtime=aws_lambda.Runtime.PYTHON_3_9,
            description="コンポーネント用のイメージを作成",
            role=lambda_role,
            environment={
                "TABLE_NAME": self._table.table_name,
                "CODEBUILD_PROJECT_NAME": self._docker_image_buildproject.project_name,
                "COMPONENT_IMAGE_REPOSITORY": self._component_ecr.repository_name,
                "COMPONENT_APP_SOURCE_REPOSITORY": self._component_source_repository.repository_clone_url_grc,
                "COMPONENT_BASE_IMAGE_REPOSITORY": self._component_base_ecr.repository_uri
            }
        )
        self._table.grant_read_write_data(lambdaFn)

        return lambdaFn 
    
    def create_lambda_check_image_status(self) -> Resource:
        """dockerイメージのビルド状況を確認するLambda

        Returns:
            Resource: lambda
        """

        lambdaFn_name = self.get_lambda_name("check_image_status")
        role_name = self.get_role_name("check_image_status")

        lambda_role = aws_iam.Role(
            self,
            id=role_name,
            assumed_by =aws_iam.ServicePrincipal("lambda.amazonaws.com"),
            role_name=role_name,
            path="/service-role/",
            managed_policies=[
                aws_iam.ManagedPolicy.from_aws_managed_policy_name("service-role/AWSLambdaBasicExecutionRole")
            ]
        )
        lambda_role.attach_inline_policy(
            aws_iam.Policy(self, "AllowCodeBuildStatus",
                document=aws_iam.PolicyDocument(
                    statements=[
                        aws_iam.PolicyStatement(
                            actions=["codebuild:BatchGetBuilds"],
                            resources=[self._docker_image_buildproject.project_arn]
                        ),
                        aws_iam.PolicyStatement(
                            actions=[
                                "dynamodb:PutItem",
                                "dynamodb:GetItem",
                                "dynamodb:UpdateItem"
                            ],
                            resources=[self._table.table_arn]
                        )
                    ]
                )
            )
        )

        lambdaFn_path = self.get_lambda_path("check_image_status")
        lambdaFn = aws_lambda.Function(
            self,
            id=lambdaFn_name,
            function_name=lambdaFn_name,
            code=aws_lambda.AssetCode(path=lambdaFn_path),
            handler="lambda_handler.handler",
            timeout=Duration.seconds(10),
            runtime=aws_lambda.Runtime.PYTHON_3_9,
            description="コンポーネント用のイメージのビルド結果を確認",
            role=lambda_role,
            environment={
                "TABLE_NAME": self._table.table_name
            }
        )

        return lambdaFn
    
    def create_lambda_create_component(self) -> Resource:
        """AWS IoT Greengrass用のコンポーネントを作成するLambda

        Returns:
            Resource: lambda
        """

        lambdaFn_name = self.get_lambda_name("create_component")
        role_name = self.get_role_name("create_component")

        lambda_role = aws_iam.Role(
            self,
            id=role_name,
            assumed_by =aws_iam.ServicePrincipal("lambda.amazonaws.com"),
            role_name=role_name,
            path="/service-role/",
            managed_policies=[
                aws_iam.ManagedPolicy.from_aws_managed_policy_name("service-role/AWSLambdaBasicExecutionRole")
            ]
        )
        lambda_role.attach_inline_policy(
            aws_iam.Policy(self, "AllowComponentCreate",
                document=aws_iam.PolicyDocument(
                    statements=[
                        aws_iam.PolicyStatement(
                            actions=["greengrass:CreateComponentVersion"],
                            resources=[f"arn:aws:greengrass:{Aws.REGION}:{Aws.ACCOUNT_ID}:components:*"]
                        ),
                        aws_iam.PolicyStatement(
                            actions=["codecommit:GetFile"],
                            resources=[self._component_source_repository.repository_arn]
                        ),
                        aws_iam.PolicyStatement(
                            actions=[
                                "dynamodb:PutItem",
                                "dynamodb:GetItem",
                                "dynamodb:UpdateItem"
                            ],
                            resources=[self._table.table_arn]
                        )
                    ]
                )
            )
        )

        lambdaFn_path = self.get_lambda_path("create_component")
        lambdaFn = aws_lambda.Function(
            self,
            id=lambdaFn_name,
            function_name=lambdaFn_name,
            code=aws_lambda.AssetCode(path=lambdaFn_path),
            handler="lambda_handler.handler",
            timeout=Duration.seconds(10),
            runtime=aws_lambda.Runtime.PYTHON_3_9,
            description="AWS IoT Greengrass用の推論コンポーネントを作成",
            role=lambda_role,
            environment={
                "TABLE_NAME": self._table.table_name,
                "COMPONENT_APP_SOURCE_REPOSITORY": self._component_source_repository.repository_clone_url_grc,
            }
        )

        return lambdaFn
    
    def create_lambda_deploy_component(self) -> Resource:
        """AWS IoT Greengrass用のコンポーネントをデプロイするLambda

        Returns:
            Resource: lambda
        """

        lambdaFn_name = self.get_lambda_name("deploy_component")
        role_name = self.get_role_name("deploy_component")

        lambda_role = aws_iam.Role(
            self,
            id=role_name,
            assumed_by =aws_iam.ServicePrincipal("lambda.amazonaws.com"),
            role_name=role_name,
            path="/service-role/",
            managed_policies=[
                aws_iam.ManagedPolicy.from_aws_managed_policy_name("service-role/AWSLambdaBasicExecutionRole")
            ]
        )
        lambda_role.attach_inline_policy(
            aws_iam.Policy(self, "AllowComponentDeploy",
                document=aws_iam.PolicyDocument(
                    statements=[
                        aws_iam.PolicyStatement(
                            actions=[
                                "iot:DescribeThingGroup"
                            ],
                            resources=[
                                f"arn:aws:iot:{Aws.REGION}:{Aws.ACCOUNT_ID}:thinggroup/*"
                            ]
                        ),
                        aws_iam.PolicyStatement(
                            actions=[
                                "iot:CreateJob"
                            ],
                            resources=[
                                f"arn:aws:iot:{Aws.REGION}:{Aws.ACCOUNT_ID}:thinggroup/*",
                                f"arn:aws:iot:{Aws.REGION}:{Aws.ACCOUNT_ID}:job/*"
                            ]
                        ),
                        aws_iam.PolicyStatement(
                            actions=[
                                "iot:DescribeJob",
                                "iot:CancelJob"
                            ],
                            resources=[
                                f"arn:aws:iot:{Aws.REGION}:{Aws.ACCOUNT_ID}:job/*"
                            ]
                        ),
                        aws_iam.PolicyStatement(
                            actions=[
                                "greengrass:CreateDeployment",
                                "greengrass:GetDeployment"
                            ],
                            resources=[
                                f"arn:aws:greengrass:{Aws.REGION}:{Aws.ACCOUNT_ID}:deployments",
                                f"arn:aws:greengrass:{Aws.REGION}:{Aws.ACCOUNT_ID}:deployments:*"
                            ]
                        ),
                        aws_iam.PolicyStatement(
                            actions=[
                                "dynamodb:PutItem",
                                "dynamodb:GetItem",
                                "dynamodb:UpdateItem"
                            ],
                            resources=[self._table.table_arn]
                        )
                    ]
                )
            )
        )

        lambdaFn_path = self.get_lambda_path("deploy_component")

        lambdaFn = aws_lambda.Function(
            self,
            id=lambdaFn_name,
            function_name=lambdaFn_name,
            code=aws_lambda.AssetCode(path=lambdaFn_path),
            handler="lambda_handler.handler",
            timeout=Duration.seconds(300),
            runtime=aws_lambda.Runtime.PYTHON_3_9,
            description="AWS IoT Greengrassへのデプロイ結果を取得",
            role=lambda_role,
            environment={
                "TABLE_NAME": self._table.table_name
            }
        )

        return lambdaFn
    
    def create_lambda_check_deploy_status(self) -> Resource:
        """AWS IoT Greengrass用のコンポーネントをデプロイするLambda

        Returns:
            Resource: lambda
        """

        lambdaFn_name = self.get_lambda_name("component_deploy_status")
        role_name = self.get_role_name("component_deploy_status")

        lambda_role = aws_iam.Role(
            self,
            id=role_name,
            assumed_by =aws_iam.ServicePrincipal("lambda.amazonaws.com"),
            role_name=role_name,
            path="/service-role/",
            managed_policies=[
                aws_iam.ManagedPolicy.from_aws_managed_policy_name("service-role/AWSLambdaBasicExecutionRole")
            ]
        )
        lambda_role.attach_inline_policy(
            aws_iam.Policy(self, "AllowComponentDeployCheck",
                document=aws_iam.PolicyDocument(
                    statements=[
                        aws_iam.PolicyStatement(
                            actions=["greengrass:GetDeployment"],
                            resources=[f"arn:aws:greengrass:{Aws.REGION}:{Aws.ACCOUNT_ID}:deployments:*"]
                        ),
                        aws_iam.PolicyStatement(
                            actions=[
                                "iot:DescribeThingGroup"
                            ],
                            resources=[
                                f"arn:aws:iot:{Aws.REGION}:{Aws.ACCOUNT_ID}:thinggroup/*"
                            ]
                        ),
                        aws_iam.PolicyStatement(
                            actions=[
                                "iot:ListThingsInThingGroup"
                            ],
                            resources=["*"]
                        ),
                        aws_iam.PolicyStatement(
                            actions=[
                                "iot:DescribeJob",
                                "iot:DescribeJobExecution"
                            ],
                            resources=[
                                f"arn:aws:iot:{Aws.REGION}:{Aws.ACCOUNT_ID}:job/*"
                            ]
                        ),
                        aws_iam.PolicyStatement(
                            actions=[
                                "dynamodb:PutItem",
                                "dynamodb:GetItem",
                                "dynamodb:UpdateItem"
                            ],
                            resources=[self._table.table_arn]
                        )
                    ]
                )
            )
        )

        lambdaFn_path = self.get_lambda_path("deploy_status")

        lambdaFn = aws_lambda.Function(
            self,
            id=lambdaFn_name,
            function_name=lambdaFn_name,
            code=aws_lambda.AssetCode(path=lambdaFn_path),
            handler="lambda_handler.handler",
            timeout=Duration.seconds(300),
            runtime=aws_lambda.Runtime.PYTHON_3_9,
            description="AWS IoT Greengrassへのデプロイ結果を取得",
            role=lambda_role,
            environment={
                "TABLE_NAME": self._table.table_name
            }
        )

        return lambdaFn

    def create_stepfunction(self) -> Resource:
        """コンポーネントをビルドしてデプロイするステートマシンの作成

        Returns:
            Resource: step function
        """

        name = f"{self.stack_name}_{self.component_id}_edgedeploy_pipeline"
        role_name = self.get_role_name("edgedeploy_pipeline")

        sf_role = aws_iam.Role(
            self,
            id=role_name,
            assumed_by =aws_iam.ServicePrincipal("states.amazonaws.com"),
            role_name=role_name,
            path="/service-role/",
            managed_policies=[
                aws_iam.ManagedPolicy.from_aws_managed_policy_name("service-role/AWSLambdaBasicExecutionRole")
            ]
        )
        sf_role.attach_inline_policy(
            aws_iam.Policy(self, "AllowCloudWatchLogsForSF",
                document=aws_iam.PolicyDocument(
                    statements=[
                        aws_iam.PolicyStatement(
                            actions=[
                                "logs:CreateLogDelivery",
                                "logs:GetLogDelivery",
                                "logs:UpdateLogDelivery",
                                "logs:DeleteLogDelivery",
                                "logs:ListLogDeliveries",
                                "logs:PutResourcePolicy",
                                "logs:DescribeResourcePolicies",
                                "logs:DescribeLogGroups"
                            ],
                            resources=["*"]
                        )
                    ]
                )
            )
        )
        sf_role.attach_inline_policy(
            aws_iam.Policy(self, "AllowXRayForSF",
                document=aws_iam.PolicyDocument(
                    statements=[
                        aws_iam.PolicyStatement(
                            actions=[
                                "xray:PutTraceSegments",
                                "xray:PutTelemetryRecords",
                                "xray:GetSamplingRules",
                                "xray:GetSamplingTargets"
                            ],
                            resources=["*"]
                        )
                    ]
                )
            )
        )
        sf_role.attach_inline_policy(
            aws_iam.Policy(self, "AllowInvokeLambda",
                document=aws_iam.PolicyDocument(
                    statements=[
                        aws_iam.PolicyStatement(
                            actions=["lambda:InvokeFunction"],
                            resources=[
                                self._lambda_build_image.function_arn,
                                self._lambda_build_image.function_arn + ":*",
                                self._lambda_check_image_status.function_arn,
                                self._lambda_check_image_status.function_arn + ":*",
                                self._lambda_create_component.function_arn,
                                self._lambda_create_component.function_arn + ":*",
                                self._lambda_deploy_component.function_arn,
                                self._lambda_deploy_component.function_arn + ":*",
                                self._lambda_check_deploy_status.function_arn,
                                self._lambda_check_deploy_status.function_arn + ":*"
                            ]
                        )
                    ]
                )
            )
        )

        # dockerコンテナをビルド
        task_build_image = aws_sf_tasks.LambdaInvoke(self,
            "BuildInferenceImage",
            lambda_function=self._lambda_build_image,
            output_path="$.Payload"
        )
        # dockerコンテナのビルド結果を確認
        task_check_build_image_status = aws_sf_tasks.LambdaInvoke(self,
            "CheckDockerImageBuildStatus",
            lambda_function=self._lambda_check_image_status,
            output_path="$.Payload"
        )

        # dockerコンテナのビルドを待つ
        wait_image_build = aws_sf.Wait(self, 
            "WaitImageBuildFinish",
            time=aws_sf.WaitTime.duration(Duration.seconds(30))
        )

        # Greengrassのコンポーネントを作成
        task_create_greengrass_component = aws_sf_tasks.LambdaInvoke(self,
            "CreateComponent",
            lambda_function=self._lambda_create_component,
            output_path="$.Payload"
        )

        # Greengrassへデプロイ
        task_deploy_component = aws_sf_tasks.LambdaInvoke(self,
            "DeployComponent",
            lambda_function=self._lambda_deploy_component,
            output_path="$.Payload"
        )

        # Greengrassへのデプロイ終了を待つ
        wait_component_deploy = aws_sf.Wait(self, 
            "WaitDeploymentFinish",
            time=aws_sf.WaitTime.duration(Duration.seconds(30))
        )

        # Greengrassへデプロイ結果を確認
        task_check_deployment_status = aws_sf_tasks.LambdaInvoke(self,
            "CheckDeploymentStatus",
            lambda_function=self._lambda_check_deploy_status,
            output_path="$.Payload"
        )

        # デプロイ失敗
        pipeline_failed = aws_sf.Fail(self, "PipelineFailed",
            error="DeployPipelineFailed",
            cause="Something went wrong"
        )
        # 正常終了
        pipeline_success = aws_sf.Succeed(self, "PipelineSuccessed")

        # dockerコンテナが存在したかを判定
        choice_component_exists_result = aws_sf.Choice(self, "JudgeComponentExists")

        # dockerコンテナのビルド結果を判定
        choice_image_build_result = aws_sf.Choice(self, "JudgeImageBuildStatus")

        # dockerコンテナのビルド結果を判定
        choice_deployment_result = aws_sf.Choice(self, "JudgeDeploymentStatus")

        # 正常終了を通知
        publish_success_message = aws_sf_tasks.SnsPublish(self,
            "Publish Success message",
            topic=aws_sns.Topic(self, "SendDeploySuccess"),
            message=aws_sf.TaskInput.from_json_path_at("$.message")
        ).next(pipeline_success)

        # デプロイ失敗を通知
        publish_failed_message = aws_sf_tasks.SnsPublish(self,
            "Publish Failed message",
            topic=aws_sns.Topic(self, "SendPipelineFailed"),
            message=aws_sf.TaskInput.from_json_path_at("$.message")
        ).next(pipeline_failed)

        definition = \
            task_build_image.next(
                choice_component_exists_result
                    .when(
                        aws_sf.Condition.string_equals("$.status", "component_exists"), task_deploy_component)
                    .otherwise(
                        wait_image_build.next(
                        task_check_build_image_status).next(
                            choice_image_build_result.when(
                                aws_sf.Condition.string_equals("$.status", "image_exists"), task_create_greengrass_component
                                .next(task_deploy_component)
                                .next(wait_component_deploy)
                                .next(task_check_deployment_status)
                                .next(
                                    choice_deployment_result
                                        .when(aws_sf.Condition.string_equals("$.status", "RUNNING"), wait_component_deploy)
                                        .when(aws_sf.Condition.string_equals("$.status", "COMPLETED"), publish_success_message)
                                        .otherwise(publish_failed_message).afterwards()))
                            .when(
                                aws_sf.Condition.string_equals("$.status", "image_faild"), publish_failed_message)
                            .otherwise(
                                wait_image_build).afterwards())
                )
            )
            #.next(aws_sf.Succeed(self, "GreengrassComponentDeployFinished"))

        state_machine = aws_sf.StateMachine(
            self,
            id=name,
            state_machine_name=name,
            definition=definition,
            state_machine_type=aws_sf.StateMachineType.STANDARD,
            role=sf_role
        )

    def create_deploy_pipeline(self) -> Resource:
        """
        AWS IoT Greengrass v2 用の推論コンポーネントを作成してデプロイするパイプラインを作成
        """

        self._table = self.create_deploy_status_table()

        self._component_source_repository = self.create_component_source_repository()
        self._component_base_ecr = self.create_ecr_component_base_repository()
        self._component_ecr = self.create_ecr_component_repository()
        self._docker_image_buildproject = self.create_docker_image_buildproject()

        self._lambda_build_image = self.create_lambda_build_image()
        self._lambda_check_image_status = self.create_lambda_check_image_status()
        self._lambda_create_component = self.create_lambda_create_component()
        self._lambda_deploy_component = self.create_lambda_deploy_component()
        self._lambda_check_deploy_status = self.create_lambda_check_deploy_status()

        self._stepfunction = self.create_stepfunction()