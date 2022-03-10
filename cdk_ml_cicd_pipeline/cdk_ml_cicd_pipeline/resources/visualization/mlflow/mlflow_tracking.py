from aws_cdk import CfnOutput, Duration, RemovalPolicy
from aws_cdk import aws_ec2 as ec2
from aws_cdk import aws_ecs as ecs
from aws_cdk import aws_ecs_patterns as ecs_patterns
from aws_cdk import aws_iam as iam
from aws_cdk import aws_rds as rds
from aws_cdk import aws_s3 as s3
from aws_cdk import aws_secretsmanager as sm
from constructs import Construct


class MLflowTracking(Construct):

    def __init__(self, scope: Construct, stack_name: str, component_id: str, **kwargs) -> None:
        super().__init__(scope=scope, id=component_id, **kwargs)

        self.stack_name = stack_name
        self.component_id = component_id

        self.dbname = "mlflowdb"
        self.port = 3306
        self.username = "master"
        self.bucket_name = f"{self.stack_name}-{self.component_id}-artifacts-store".lower()
        self.container_repo_name = "mlflow-containers"
        self.cluster_name = "mlflow"
        self.service_name = "mlflow"

        # Create Resources
        self._create_im_role()
        self._create_secret()
        self._create_network()
        self._create_artifact_store()
        self._create_backend_store()
        self._create_mlflow_server()
        self._create_outputs()

    def _create_im_role(self):
        """
        Create IAM Role for ECS
        """
        task_role_id = f"{self.stack_name}-{self.component_id}-task-role"
        self.role = iam.Role(
            scope=self, id=task_role_id, assumed_by=iam.ServicePrincipal(service="ecs-tasks.amazonaws.com")
        )
        self.role.add_managed_policy(iam.ManagedPolicy.from_aws_managed_policy_name("AmazonS3FullAccess"))
        self.role.add_managed_policy(iam.ManagedPolicy.from_aws_managed_policy_name("AmazonECS_FullAccess"))

    def _create_secret(self):
        """
        Create a secret for RDS
        """
        db_password_secret_id = f"{self.stack_name}-{self.component_id}-db_password_secret"
        secret_name = f"{self.stack_name}-{self.component_id}-dbPassword"
        self.db_password_secret = sm.Secret(
            scope=self,
            id=db_password_secret_id,
            secret_name=secret_name,
            generate_secret_string=sm.SecretStringGenerator(password_length=20, exclude_punctuation=True),
        )

    def _create_network(self):
        """
        Create a VPC network
        """
        public_subnet = ec2.SubnetConfiguration(name="Public", subnet_type=ec2.SubnetType.PUBLIC, cidr_mask=28)
        private_subnet = ec2.SubnetConfiguration(name="Private", subnet_type=ec2.SubnetType.PRIVATE_WITH_NAT, cidr_mask=28)
        isolated_subnet = ec2.SubnetConfiguration(name="DB", subnet_type=ec2.SubnetType.PRIVATE_ISOLATED, cidr_mask=28)
        vpc_id = f"{self.stack_name}-{self.component_id}-vpc"

        self.vpc = ec2.Vpc(
            scope=self,
            id=vpc_id,
            cidr="10.0.0.0/24",
            max_azs=2,
            nat_gateway_provider=ec2.NatProvider.gateway(),
            nat_gateways=1,
            subnet_configuration=[public_subnet, private_subnet, isolated_subnet],
        )
        self.vpc.add_gateway_endpoint("S3Endpoint", service=ec2.GatewayVpcEndpointAwsService.S3)

    def _create_artifact_store(self):
        """
        Create a S3 Bucket as a artifact store for MLflow server
        """
        artifacts_store_id = f"{self.stack_name}-{self.component_id}-artifacts-store"

        self.artifact_bucket = s3.Bucket(
            scope=self,
            id=artifacts_store_id,
            bucket_name=self.bucket_name,
            public_read_access=False,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            removal_policy=RemovalPolicy.DESTROY,
        )

    def _create_backend_store(self):
        """
        Create a RDS as a backend store for MLflow server
        """
        # Creates a security group for AWS RDS
        self.sg_rds = ec2.SecurityGroup(scope=self, id="SGRDS", vpc=self.vpc, security_group_name="sg_rds")
        # Adds an ingress rule which allows resources in the VPC's CIDR to access the database.
        self.sg_rds.add_ingress_rule(peer=ec2.Peer.ipv4("10.0.0.0/24"), connection=ec2.Port.tcp(self.port))

        backend_store_id = f"{self.stack_name}-{self.component_id}-backend-store"

        self.database = rds.DatabaseInstance(
            scope=self,
            id=backend_store_id,
            database_name=self.dbname,
            port=self.port,
            credentials=rds.Credentials.from_username(
                username=self.username, password=self.db_password_secret.secret_value
            ),
            engine=rds.DatabaseInstanceEngine.mysql(version=rds.MysqlEngineVersion.VER_8_0_19),
            instance_type=ec2.InstanceType.of(ec2.InstanceClass.BURSTABLE2, ec2.InstanceSize.SMALL),
            vpc=self.vpc,
            security_groups=[self.sg_rds],
            vpc_subnets=ec2.SubnetSelection(subnet_type=ec2.SubnetType.PRIVATE_ISOLATED),
            # multi_az=True,
            removal_policy=RemovalPolicy.DESTROY,
            deletion_protection=False,
        )

    def _create_mlflow_server(self):
        """
        Create a Farget task for MLflow server
        """
        cluster = ecs.Cluster(scope=self, id="CLUSTER", cluster_name=self.cluster_name, vpc=self.vpc)

        task_id = f"{self.stack_name}-{self.component_id}-MLflow"
        task_definition = ecs.FargateTaskDefinition(
            scope=self,
            id=task_id,
            task_role=self.role,
        )

        container_id = f"{self.stack_name}-{self.component_id}-container"
        container = task_definition.add_container(
            id=container_id,
            image=ecs.ContainerImage.from_asset(
                directory="cdk_ml_cicd_pipeline/resources/visualization/mlflow/container",
            ),
            environment={
                "BUCKET": f"s3://{self.artifact_bucket.bucket_name}",
                "HOST": self.database.db_instance_endpoint_address,
                "PORT": str(self.port),
                "DATABASE": self.dbname,
                "USERNAME": self.username,
            },
            secrets={"PASSWORD": ecs.Secret.from_secrets_manager(self.db_password_secret)},
            logging=ecs.LogDriver.aws_logs(stream_prefix='mlflow')
        )
        port_mapping = ecs.PortMapping(container_port=5000, host_port=5000, protocol=ecs.Protocol.TCP)
        container.add_port_mappings(port_mapping)

        fargate_service_id = f"{self.stack_name}-{self.component_id}-" + "mlflow-fargate"
        self.fargate_service = ecs_patterns.NetworkLoadBalancedFargateService(
            scope=self,
            id=fargate_service_id,
            service_name=self.service_name,
            cluster=cluster,
            task_definition=task_definition,
        )

        # Setup security group
        self.fargate_service.service.connections.security_groups[0].add_ingress_rule(
            peer=ec2.Peer.ipv4(self.vpc.vpc_cidr_block),
            connection=ec2.Port.tcp(5000),
            description="Allow inbound from VPC for mlflow",
        )

        # Setup autoscaling policy
        autoscaling_policy_id = f"{self.stack_name}-{self.component_id}-" + "autoscaling-policy"
        scaling = self.fargate_service.service.auto_scale_task_count(max_capacity=2)
        scaling.scale_on_cpu_utilization(
            id=autoscaling_policy_id,
            target_utilization_percent=70,
            scale_in_cooldown=Duration.seconds(60),
            scale_out_cooldown=Duration.seconds(60),
        )

    def _create_outputs(self):
        output_id = f"{self.stack_name}-{self.component_id}-" + "LoadBalancerDNS"
        CfnOutput(scope=self, id=output_id, value=self.fargate_service.load_balancer.load_balancer_dns_name)
