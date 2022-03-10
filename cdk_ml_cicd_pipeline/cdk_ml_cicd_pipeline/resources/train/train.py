import pathlib

from aws_cdk import CfnOutput, Duration, RemovalPolicy, Resource, Stack
from aws_cdk import aws_codebuild as codebuild
from aws_cdk import aws_codepipeline as codepipeline
from aws_cdk import aws_codepipeline_actions as codepipeline_actions
from aws_cdk import aws_ec2 as ec2
from aws_cdk import aws_iam as iam
from aws_cdk import aws_lambda
from aws_cdk import aws_lambda_event_sources as lambda_event_sources
from aws_cdk import aws_lambda_python_alpha as lambda_python
from aws_cdk import aws_rds as rds
from aws_cdk import aws_s3 as s3
from aws_cdk import aws_sns as sns
from aws_cdk import aws_stepfunctions as stepfunctions
from constructs import Construct


class Train(Construct):
    @property
    def handler(self) -> Resource:
        return self._handler

    def __init__(self, scope: Construct, stack_name: str, component_id: str, **kwargs):
        super().__init__(scope=scope, id=component_id, **kwargs)

        self.context = self.node.try_get_context('train')
        self.region = self.node.try_get_context('region')
        self.name_prefix = f"{stack_name}-{component_id}"
        self.sfn_name = f"{self.name_prefix}_{self.context['stepfunctions_name']}"

        self.source_output = codepipeline.Artifact()


        # Create Resources
        self._create_s3bucket()
        self._create_roles_for_train_step()
        self._create_roles_for_set_experiment_info_env_step()

        self.sns_topic = self._create_slack_notify_sns_topic()
        self._create_network()
        self._create_rds()
        self.set_experiment_info_env_lambdaFn = self._create_lambda_for_set_experiment_info_env()
        self._create_lambda_for_manual_approval()
        self.postprocess_lambdaFn = self._create_lambda_for_post_process()

        self._create_pipeline()
        self._create_lambda()
        
        self._cfnoutput()


    def _create_source(self):
        action = codepipeline_actions.CodeStarConnectionsSourceAction(
            action_name=f"{self.name_prefix}-source-action",
            owner=self.context['owner'],
            repo=self.context['repo'],
            output=self.source_output,
            connection_arn=self.context['connection'],
            branch=self.context['branch']
        )
        self.pipeline.add_stage(stage_name='Source', actions=[action])

    
    def _create_train_step(self):
        stage = self.pipeline.add_stage(stage_name=f"{self.name_prefix}-stage")

        role = iam.Role(self, "Role",
            assumed_by=iam.ServicePrincipal("codebuild.amazonaws.com"),
            description="Role for CodeBuild",
            role_name=f"{self.name_prefix}-codebuild-role",
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name("AmazonEC2ContainerRegistryFullAccess"),
                iam.ManagedPolicy.from_aws_managed_policy_name("AWSStepFunctionsFullAccess"),
                iam.ManagedPolicy.from_aws_managed_policy_name("service-role/AWSLambdaVPCAccessExecutionRole"),
                iam.ManagedPolicy.from_aws_managed_policy_name("SecretsManagerReadWrite"),
            ],
        )

        policy = iam.Policy(self, "PassRolePolicy")
        policy.document.add_statements(iam.PolicyStatement(
            actions=["iam:PassRole"],
            resources=[f"arn:aws:iam::{Stack.of(self).account}:role/*"]
        ))
        role.attach_inline_policy(policy)

        build_spec = codebuild.BuildSpec.from_source_filename('buildspec.yml')
        project = codebuild.PipelineProject(self, "TrainingStepProject",
            build_spec=build_spec,
            environment=codebuild.BuildEnvironment(
                build_image=codebuild.LinuxBuildImage.STANDARD_5_0,
                privileged=True
            ),
            role=role,
            security_groups=[self.security_group],
            subnet_selection=self.subnet_selection,
            vpc=self.vpc
        )

        action = codepipeline_actions.CodeBuildAction(
            action_name=f"{self.name_prefix}-training-action",
            project=project,
            input=self.source_output,
            environment_variables={
                "EXEC_ID": codebuild.BuildEnvironmentVariable(value='#{codepipeline.PipelineExecutionId}'),
                "SFN_WORKFLOW_NAME": codebuild.BuildEnvironmentVariable(value=self.sfn_name)
            },
            variables_namespace="trainStep",
        )
        stage.add_action(action)


    def _create_stepfunctions_step(self):
        state_machine_arn = f'arn:aws:states:{self.region}:{Stack.of(self).account}:stateMachine:{self.sfn_name}'
        stage = self.pipeline.add_stage(stage_name="StepFunctions")
        params={
            'TrainingJobName': '#{trainStep.TRAIN_JOB_NAME}',
            'EvaluationJobName': '#{trainStep.EVAL_JOB_NAME}',
            'PreprocessingJobName': '#{trainStep.PREP_JOB_NAME}'
        }
        action = codepipeline_actions.StepFunctionInvokeAction(
            action_name=f"{self.name_prefix}-stepfunctions-action",
            state_machine=stepfunctions.StateMachine.from_state_machine_arn(self, "StepFunctions",
                state_machine_arn=state_machine_arn
            ),
            state_machine_input=codepipeline_actions.StateMachineInput.literal(params)
        )
        stage.add_action(action)


    def _create_set_experiment_info_env_step(self) -> Resource:
        stage = self.pipeline.add_stage(stage_name="SetExperimentInfoEnvs")
        action = codepipeline_actions.LambdaInvokeAction(
            action_name=f"{self.name_prefix}-set-experiment-info-env-action",
            user_parameters={
                'EVAL_REPORT_S3': '#{trainStep.EVAL_REPORT_S3}',
            },
            lambda_=self.set_experiment_info_env_lambdaFn,
            variables_namespace="experimentInfo",
        )

        stage.add_action(action)    


    def _create_manual_approve_step(self):
        # MLflow のダッシュボードを見るための情報
        mlflow_tracking_url = '#{trainStep.MLFLOW_SERVER_URI}'
        # EXPERIMENTS_ID = "4"
        experiment_name = '#{experimentInfo.experiment_name}'
        experiment_id = '#{experimentInfo.experiment_id}'
        # RUN_ID = "11ab290e535140609660b5d894ccdf17"
        run_id = '#{experimentInfo.run_id}'
        topic = self.sns_topic

        additional_info = f"Experiment name: {experiment_name} \nRun id: {run_id} "

        # mlflow_link = mlflow_tracking_url + "#/experiments/" + EXPERIMENTS_ID + "/runs/" + RUN_ID
        mlflow_link = mlflow_tracking_url + "#/experiments/" + experiment_id + "/runs/" + run_id

        stage = self.pipeline.add_stage(stage_name="Approve")
        action = codepipeline_actions.ManualApprovalAction(
            action_name=f"{self.name_prefix}-approval-action",
            notification_topic=topic,
            external_entity_link=mlflow_link,
            additional_information=additional_info
        )
        stage.add_action(action)
        

    def _create_post_process_step(self) -> Resource:
        stage = self.pipeline.add_stage(stage_name="PostProcess")

        action = codepipeline_actions.LambdaInvokeAction(
            action_name=f"{self.name_prefix}-postprocess-action",
            user_parameters={
                'EXPERIMENT_NAME' : '#{experimentInfo.experiment_name}',
                'RUN_ID': '#{experimentInfo.run_id}',
                'TRAIN_JOB_NAME': '#{trainStep.TRAIN_JOB_NAME}',
                'TRAINED_MODEL_S3': '#{trainStep.TRAINED_MODEL_S3}',
            },
            lambda_=self.postprocess_lambdaFn
        )

        stage.add_action(action)


    def _create_pipeline(self) -> Resource:
        """
        Create CodePipeline for training.
        """

        self.pipeline = codepipeline.Pipeline(self, "Pipeline")

        self._create_source()
        self._create_train_step()
        self._create_stepfunctions_step()
        self._create_set_experiment_info_env_step()
        self._create_manual_approve_step()
        self._create_post_process_step()


    def _create_slack_notify_sns_topic(self) -> Resource:
        """
        Create SNS topic for slack notification.
        """
        topic_id = f"{self.name_prefix}-sns_slack_notify_topic"
        topic = sns.Topic(scope=self, id=topic_id, fifo=False, topic_name=f"{self.name_prefix}-slack_notify")
        return topic


    def _create_lambda_for_set_experiment_info_env(self) -> Resource:
        """
        Ref: https://github.com/aws-samples/aws-cdk-examples/tree/master/python/lambda-cron
        """
        lambdaFn_id = f"{self.name_prefix}-set-experiment-info-env-lambda_handler"
        entry = str(pathlib.Path(__file__).resolve().parent) + "/lambdafn_set_experiment_info_env/"

        lambdaFn = lambda_python.PythonFunction(
            scope=self,
            id=lambdaFn_id,
            entry=entry,
            index="lambda_handler.py",
            handler="lambda_handler",
            timeout=Duration.seconds(300),
            runtime=aws_lambda.Runtime.PYTHON_3_8,
            role=self.lambda_experiment_info_role
        )

        return lambdaFn


    def _create_lambda_for_manual_approval(self) -> Resource:
        """
        Ref: https://github.com/aws-samples/aws-cdk-examples/tree/master/python/lambda-cron
        """
        lambdaFn_id = f"{self.name_prefix}-manual-approval-lambda_handler"
        entry = str(pathlib.Path(__file__).resolve().parent) + "/lambdafn_manual_approve/"
        topic = self.sns_topic
        slack_hook_url = self.context["slack_hook_url"]

        lambdaFn = lambda_python.PythonFunction(
            scope=self,
            id=lambdaFn_id,
            entry=entry,
            index="lambda_handler.py",
            handler="lambda_handler",
            timeout=Duration.seconds(300),
            runtime=aws_lambda.Runtime.PYTHON_3_8,
            environment={
                "slack_hook_url": slack_hook_url
            }
        )

        sns_event_source = lambda_event_sources.SnsEventSource(topic)
        lambdaFn.add_event_source(sns_event_source)

        return lambdaFn

    
    def _create_lambda_for_post_process(self) -> Resource:
        """
        Ref: https://github.com/aws-samples/aws-cdk-examples/tree/master/python/lambda-cron
        """
        lambdaFn_id = f"{self.name_prefix}-post-process-lambda_handler"
        entry = str(pathlib.Path(__file__).resolve().parent) + "/lambdafn_post_process/"
        slack_hook_url = self.context["slack_hook_url"]

        lambdaFn = lambda_python.PythonFunction(
            scope=self,
            id=lambdaFn_id,
            entry=entry,
            index="lambda_handler.py",
            handler="lambda_handler",
            timeout=Duration.seconds(300),
            runtime=aws_lambda.Runtime.PYTHON_3_8,
            environment={
                "slack_hook_url": slack_hook_url
            }
        )

        return lambdaFn


    def _create_s3bucket(self):
        """
        Create S3 Bucket
        """
        
        bucket_name=f"sagemaker-{self.region}-{self.name_prefix}"
        self.s3_bucket = s3.Bucket(self, "Bucket", removal_policy=RemovalPolicy.DESTROY,bucket_name=bucket_name.lower())


    def _create_roles_for_train_step(self):
        """
        Create Roles for Train Step
        """        

        self.sagemaker_exec_role = iam.Role(self, "SageMakerExecRole",
            assumed_by=iam.ServicePrincipal("sagemaker.amazonaws.com"),
            description="Role for SageMaker Exec",
            role_name=f"{self.name_prefix}-sagemaker-exec-role",
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name("AmazonEC2ContainerRegistryFullAccess"),
                iam.ManagedPolicy.from_aws_managed_policy_name("AmazonSageMakerFullAccess")
            ],
        )

        sagemaker_policy = iam.Policy(self, "SageMakerPolicy")
        sagemaker_policy.document.add_statements(iam.PolicyStatement(
            effect=iam.Effect.ALLOW,
            actions=[
                "s3:GetObject",
                "s3:PutObject",
                "s3:DeleteObject",
                "s3:ListBucket"
            ],
            resources=[f"arn:aws:s3:::*"]
        ))
        self.sagemaker_exec_role.attach_inline_policy(sagemaker_policy)     

        self.sfn_wf_exec_role = iam.Role(self, "SFnWfExecRole",
            assumed_by=iam.ServicePrincipal("states.amazonaws.com"),
            description="Role for SFn Workflow Exec",
            role_name=f"{self.name_prefix}-sfn-wf-exec-role",
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name("CloudWatchEventsFullAccess"),
            ],
        )

        sfn_wf_exec_passrole_policy = iam.Policy(self, "SFnWfExecPassRolePolicy")
        sfn_wf_exec_passrole_policy.document.add_statements(iam.PolicyStatement(
            actions=["iam:PassRole"],
            resources=[self.sagemaker_exec_role.role_arn]
        ))
        self.sfn_wf_exec_role.attach_inline_policy(sfn_wf_exec_passrole_policy) 

        sfn_wf_exec_event_policy = iam.Policy(self, "SFnWfExecEventPolicy")
        sfn_wf_exec_event_policy.document.add_statements(iam.PolicyStatement(
            effect=iam.Effect.ALLOW,
            actions=[
                "events:PutTargets",
                "events:DescribeRule",
                "events:PutRule"
            ],
            resources=[
                "arn:aws:events:*:*:rule/StepFunctionsGetEventsForSageMakerTrainingJobsRule",
                "arn:aws:events:*:*:rule/StepFunctionsGetEventsForSageMakerTransformJobsRule",
                "arn:aws:events:*:*:rule/StepFunctionsGetEventsForSageMakerTuningJobsRule",
                "arn:aws:events:*:*:rule/StepFunctionsGetEventsForECSTaskRule",
                "arn:aws:events:*:*:rule/StepFunctionsGetEventsForBatchJobsRule"
            ]
        ))
        self.sfn_wf_exec_role.attach_inline_policy(sfn_wf_exec_event_policy)

        sfn_wf_exec_policy = iam.Policy(self, "SFnWfExecPolicy")
        sfn_wf_exec_policy.document.add_statements(iam.PolicyStatement(
            effect=iam.Effect.ALLOW,
            actions=[
                "batch:DescribeJobs",
                "batch:SubmitJob",
                "batch:TerminateJob",
                "dynamodb:DeleteItem",
                "dynamodb:GetItem",
                "dynamodb:PutItem",
                "dynamodb:UpdateItem",
                "ecs:DescribeTasks",
                "ecs:RunTask",
                "ecs:StopTask",
                "glue:BatchStopJobRun",
                "glue:GetJobRun",
                "glue:GetJobRuns",
                "glue:StartJobRun",
                "lambda:InvokeFunction",
                "sagemaker:CreateEndpoint",
                "sagemaker:CreateEndpointConfig",
                "sagemaker:CreateHyperParameterTuningJob",
                "sagemaker:CreateModel",
                "sagemaker:CreateProcessingJob",
                "sagemaker:CreateTrainingJob",
                "sagemaker:CreateTransformJob",
                "sagemaker:DeleteEndpoint",
                "sagemaker:DeleteEndpointConfig",
                "sagemaker:DescribeHyperParameterTuningJob",
                "sagemaker:DescribeProcessingJob",
                "sagemaker:DescribeTrainingJob",
                "sagemaker:DescribeTransformJob",
                "sagemaker:ListProcessingJobs",
                "sagemaker:ListTags",
                "sagemaker:StopHyperParameterTuningJob",
                "sagemaker:StopProcessingJob",
                "sagemaker:StopTrainingJob",
                "sagemaker:StopTransformJob",
                "sagemaker:UpdateEndpoint",
                "sns:Publish",
                "sqs:SendMessage"
            ],
            resources=["*"]
        ))
        self.sfn_wf_exec_role.attach_inline_policy(sfn_wf_exec_policy)


    def _create_roles_for_set_experiment_info_env_step(self):
        """
        Create Roles for Set Experiment Info Env Step
        """

        self.lambda_experiment_info_role = iam.Role(self, "LambdaExperimentInfoRole",
            assumed_by=iam.ServicePrincipal("lambda.amazonaws.com"),
            description="Role for Lambda to get experiments info at S3.",
            role_name=f"{self.name_prefix}-lambda-experiment-info-role",
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name("AmazonS3ReadOnlyAccess")
            ],
        )

        lambda_experiment_info_policy = iam.Policy(self, "LambdaExperimentInfoPolicy")
        lambda_experiment_info_policy.add_statements(iam.PolicyStatement(
            effect=iam.Effect.ALLOW,
            actions=[
                "logs:CreateLogGroup",
                "logs:CreateLogStream",
                "logs:PutLogEvents"
            ],
            resources=["*"]
        ))
        self.lambda_experiment_info_role.attach_inline_policy(lambda_experiment_info_policy)


    def _create_network(self):
        """
        Create VPC, PrivateSubnet, SecrityGroup for RDS
        """

        public_subnet_config = ec2.SubnetConfiguration(
            cidr_mask=24,
            name=f"{self.name_prefix}-subnet-public",
            subnet_type=ec2.SubnetType.PUBLIC
        )
        private_subnet_config = ec2.SubnetConfiguration(
            cidr_mask=24,
            name=f"{self.name_prefix}-subnet-private",
            subnet_type=ec2.SubnetType.PRIVATE_WITH_NAT
        )
        self.vpc = ec2.Vpc(self, "VPC",
            cidr=self.context['vpc']['cidr'],
            subnet_configuration=[public_subnet_config, private_subnet_config],
            nat_gateways=1,
            max_azs=2
        )
        self.security_group = ec2.SecurityGroup(self, "SecurityGroupForRDS",
            vpc=self.vpc,
            security_group_name=f"{self.name_prefix}-security-group-name"
        )
        
        self.security_group.add_ingress_rule(self.security_group, 
            ec2.Port.tcp(3306), "allow access from my security group")
    
        
        self.subnet_selection = ec2.SubnetSelection(
            one_per_az=False,
            subnets=self.vpc.private_subnets,
        )


    
    def _create_rds(self):
        """
        Create RDS for training.
        """
        subnet_group = rds.SubnetGroup(self, "RdsSubnetGroup",
            description="for RDS",
            vpc=self.vpc,

            removal_policy=RemovalPolicy.DESTROY,
            subnet_group_name=f"{self.name_prefix}-sg-rds",
            vpc_subnets=self.subnet_selection
        )
        
        self.rds = rds.DatabaseInstance(self, f"{self.name_prefix}-database",
            engine=rds.DatabaseInstanceEngine.mysql(
                version=rds.MysqlEngineVersion.VER_8_0_16
            ),
            vpc=self.vpc,
            port=3306,
            instance_type=ec2.InstanceType.of(
                ec2.InstanceClass["MEMORY4"],
                ec2.InstanceSize["LARGE"] 
            ),
            removal_policy=RemovalPolicy.DESTROY,
            deletion_protection=False,
            security_groups=[self.security_group],
            subnet_group=subnet_group
        )


    def _create_lambda(self):
        role = iam.Role(self, "LambdaPrepareDbRole",
            assumed_by=iam.ServicePrincipal("lambda.amazonaws.com"),
            description="Role for Lambda preparing RDS",
            role_name=f"{self.name_prefix}-lambda-prepare-db-role",
            managed_policies=[
                #iam.ManagedPolicy.from_aws_managed_policy_name("AWSLambdaBasicExecutionRole"),
                iam.ManagedPolicy.from_aws_managed_policy_name("service-role/AWSLambdaVPCAccessExecutionRole"),
                iam.ManagedPolicy.from_aws_managed_policy_name("SecretsManagerReadWrite"),
            ],
        )
        
        lambda_function_id = f"{self.name_prefix}-prepare_db_function"
        lambda_function_path = str(pathlib.Path(__file__).resolve().parent) + "/lambdafn/prepare_db_function/"
        lambda_layer_path = str(pathlib.Path(__file__).resolve().parent) + "/lambdafn/lambda_layer/"
        
        layer = aws_lambda.LayerVersion(self,'Layer',code=aws_lambda.AssetCode(lambda_layer_path))
        
        lambda_fn = aws_lambda.Function(
            scope=self,
            id=lambda_function_id,
            function_name=lambda_function_id,
            code=aws_lambda.AssetCode(path=lambda_function_path),
            handler="lambda_handler.lambda_handler",
            layers=[layer],
            timeout=Duration.seconds(300),
            runtime=aws_lambda.Runtime.PYTHON_3_7,
            role=role,
            description="write some description for this lambda",
            security_groups=[self.security_group],
            vpc=self.vpc,
            vpc_subnets=self.subnet_selection
        )
        
        lambda_fn.add_environment('SECRETS_NAME', self.rds.secret.secret_arn)
        lambda_fn.add_environment('REGION_NAME', self.region)


    def _cfnoutput(self):
        CfnOutput(self, f"s3Bucket", value=self.s3_bucket.bucket_name)
        CfnOutput(self, f"StepFunctionsWorkflowExecutionRole", value=self.sfn_wf_exec_role.role_arn)
        CfnOutput(self, f"AmazonSageMakerExecutionRole", value=self.sagemaker_exec_role.role_arn)
        CfnOutput(self, f"SecretsManagerArn", value=self.rds.secret.secret_arn)
        CfnOutput(self, f"StepFunctionsName", value=self.sfn_name)
