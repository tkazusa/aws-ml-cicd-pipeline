import pathlib

import aws_cdk.core as cdk
from aws_cdk import aws_codebuild as codebuild
from aws_cdk import aws_codecommit as codecommit
from aws_cdk import aws_codepipeline as codepipeline
from aws_cdk import aws_codepipeline_actions as codepipeline_actions
from aws_cdk import aws_ec2 as ec2
from aws_cdk import aws_iam as iam
from aws_cdk import aws_lambda
from aws_cdk import aws_rds as rds
from aws_cdk import core


class Train(core.Construct):
    @property
    def handler(self) -> core.Resource:
        return self._handler

    def __init__(self, scope: core.Construct, stack_name: str, component_id: str, **kwargs):
        super().__init__(scope=scope, id=component_id, **kwargs)

        self.context = self.node.try_get_context("train")
        self.region = self.node.try_get_context("region")
        self.name_prefix = f"{stack_name}-{component_id}"

        self.source_output = codepipeline.Artifact()

        # Create Resources
        self._create_network()
        self._create_rds()
        self._create_pipeline()

        self._cfnoutput()

    def _create_source(self):
        action = codepipeline_actions.CodeStarConnectionsSourceAction(
            action_name=f"{self.name_prefix}-source-action",
            owner=self.context["owner"],
            repo=self.context["repo"],
            output=self.source_output,
            connection_arn=self.context["connection"],
            branch=self.context["branch"],
        )
        self.pipeline.add_stage(stage_name="Source", actions=[action])

    def _create_stage_for_all_steps(self):
        self.stage = self.pipeline.add_stage(stage_name=f"{self.name_prefix}-stage")

    def _create_train_step(self):
        role = iam.Role(
            self,
            "Role",
            assumed_by=iam.ServicePrincipal("codebuild.amazonaws.com"),
            description="Role for CodeBuild",
            role_name=f"{self.name_prefix}-codebuild-role",
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name("AmazonEC2ContainerRegistryFullAccess"),
                iam.ManagedPolicy.from_aws_managed_policy_name("AWSStepFunctionsFullAccess"),
            ],
        )

        policy = iam.Policy(self, "PassRolePolicy")
        policy.document.add_statements(
            iam.PolicyStatement(
                actions=["iam:PassRole"], resources=[f"arn:aws:iam::{cdk.Stack.of(self).account}:role/*"]
            )
        )
        role.attach_inline_policy(policy)

        build_spec = codebuild.BuildSpec.from_source_filename("buildspec.yaml")
        project = codebuild.PipelineProject(
            self,
            "TrainingStepProject",
            build_spec=build_spec,
            environment=codebuild.BuildEnvironment(
                build_image=codebuild.LinuxBuildImage.STANDARD_5_0, privileged=True
            ),
            role=role,
            security_groups=[self.security_group],
            subnet_selection=self.subnet_selection,
            vpc=self.vpc,
        )

        action = codepipeline_actions.CodeBuildAction(
            action_name=f"{self.name_prefix}-training-action",
            project=project,
            input=self.source_output,
            environment_variables={
                "EXEC_ID": codebuild.BuildEnvironmentVariable(value="#{codepipeline.PipelineExecutionId}")
            },
        )
        self.stage.add_action(action)

    def _create_manual_approve_step(self):
        stage = self.pipeline.add_stage(stage_name="Approve")
        action = codepipeline_actions.ManualApprovalAction(
            action_name=f"{self.name_prefix}-approval-action",
        )
        stage.add_action(action)

    def _create_pipeline(self) -> core.Resource:
        """
        Create CodePipeline for training.
        """

        self.pipeline = codepipeline.Pipeline(self, "Pipeline")
        self._create_source()
        self._create_stage_for_all_steps()
        self._create_train_step()
        self._create_manual_approve_step()

    def _create_network(self):
        """
        Create VPC, PrivateSubnet, SecrityGroup for RDS
        """

        public_subnet_config = ec2.SubnetConfiguration(
            cidr_mask=24, name=f"{self.name_prefix}-subnet-public", subnet_type=ec2.SubnetType.PUBLIC
        )
        private_subnet_config = ec2.SubnetConfiguration(
            cidr_mask=24, name=f"{self.name_prefix}-subnet-private", subnet_type=ec2.SubnetType.PRIVATE
        )
        self.vpc = ec2.Vpc(
            self,
            "VPC",
            cidr=self.context["vpc"]["cidr"],
            subnet_configuration=[public_subnet_config, private_subnet_config],
            nat_gateways=1,
            max_azs=2,
        )
        self.security_group = ec2.SecurityGroup(
            self, "SecurityGroupForRDS", vpc=self.vpc, security_group_name=f"{self.name_prefix}-security-group-name"
        )

        self.subnet_selection = ec2.SubnetSelection(
            one_per_az=False,
            subnets=self.vpc.private_subnets,
        )

    def _create_rds(self):
        """
        Create RDS for training.
        """
        subnet_group = rds.SubnetGroup(
            self,
            "RdsSubnetGroup",
            description="for RDS",
            vpc=self.vpc,
            removal_policy=cdk.RemovalPolicy.DESTROY,
            subnet_group_name=f"{self.name_prefix}-sg-rds",
            vpc_subnets=self.subnet_selection,
        )

        rds.DatabaseInstance(
            self,
            f"{self.name_prefix}-database",
            engine=rds.DatabaseInstanceEngine.mysql(version=rds.MysqlEngineVersion.VER_8_0_16),
            vpc=self.vpc,
            port=3306,
            instance_type=ec2.InstanceType.of(ec2.InstanceClass["MEMORY4"], ec2.InstanceSize["LARGE"]),
            removal_policy=core.RemovalPolicy.DESTROY,
            deletion_protection=False,
            security_groups=[self.security_group],
            subnet_group=subnet_group,
        )

    def _cfnoutput(self):
        cdk.CfnOutput(self, "codestar-connections", value=self.context["connection"])
        for idx, s in enumerate(self.vpc.public_subnets):
            cdk.CfnOutput(self, f"public-subnet-{idx}", value=s.subnet_id)
        for idx, s in enumerate(self.vpc.private_subnets):
            cdk.CfnOutput(self, f"private-subnet-{idx}", value=s.subnet_id)
