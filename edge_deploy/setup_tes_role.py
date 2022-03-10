import argparse
import boto3
import botocore
import json

def arg_check() -> argparse.Namespace:
    """
    argument check
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--device_name", required=True,
                        help="[Required] AWS IoT Greengrass device name")
    parser.add_argument('--region', required=True, help='[Required] AWS Region')

    args = parser.parse_args()

    return args

def echo_setup_command(region, thing_name, tes_role_name, tes_role_alias):
    print("execute this command to setup greengrass.\n")
    print("sudo -E java -Droot='/greengrass/v2' -Dlog.store=FILE \\")
    print(f"-jar ./GreengrassCore/lib/Greengrass.jar \\")
    print(f"--aws-region {region} \\")
    print(f"--thing-name {thing_name} \\")
    print(f"--thing-group-name {thing_name}_Group \\")
    print(f"--tes-role-name {tes_role_name} \\")
    print(f"--tes-role-alias-name {tes_role_alias} \\")
    print("--component-default-user ggc_user:ggc_group \\")
    print("--provision true \\")
    print("--setup-system-service true\n")

def setup_device():
    init_info = arg_check()

    iam = boto3.client('iam')

    resource_name = f"MLOps_{init_info.device_name}"

    # Greengrass用のRoleを作成
    tes_role = f"{resource_name}_TESRole"
    tes_role_arn = None
    try:
        assume_policy = {
          "Version": "2012-10-17",
          "Statement": [
            {
              "Effect": "Allow",
              "Principal": {
                "Service": "credentials.iot.amazonaws.com"
              },
              "Action": "sts:AssumeRole"
            }
          ]
        }
        response = iam.create_role(
            RoleName=tes_role,
            AssumeRolePolicyDocument=json.dumps(assume_policy)
        )
    except iam.exceptions.EntityAlreadyExistsException as e:
      # 同じ名前のRoleがすでにある
      print(f"{tes_role} already exists")
      pass
    finally:
        response = iam.get_role(
            RoleName=tes_role
        )
        tes_role_arn = response["Role"]["Arn"]

    print(tes_role_arn)

    # Greengrassデバイスに対して許可する権限
    tes_role_policy = f"{resource_name}_TESPolicy"
    tes_role_policy_arn = None
    try:
        policy_doc = {
          "Version": "2012-10-17",
          "Statement": [
            {
              "Action": [
                "ecr:GetAuthorizationToken",
                "ecr:BatchGetImage",
                "ecr:GetDownloadUrlForLayer"
              ],
              "Resource": [
                "*"
              ],
              "Effect": "Allow"
            }
          ]
        }
        response = iam.create_policy(
            PolicyName=tes_role_policy,
            PolicyDocument=json.dumps(policy_doc)
        )
        tes_role_policy_arn = response["Policy"]["Arn"]
    except iam.exceptions.EntityAlreadyExistsException as e:
      # 同じ名前のPolicyがすでにある
      print(f"{tes_role_policy} already exists")
      tes_role_policy_arn = f"arn:aws:iam::{tes_role_arn.split(':')[4]}:policy/{tes_role_policy}"
      pass
    print(tes_role_policy_arn)

    # RoleにPolicyを紐付け
    response = iam.attach_role_policy(
        RoleName=tes_role,
        PolicyArn=tes_role_policy_arn
    )

    # Role Aliasの作成
    iot = boto3.client('iot')

    tes_role_alias = f"{tes_role}_Alias"

    try:
        response = iot.create_role_alias(
            roleAlias=tes_role_alias,
            roleArn=tes_role_arn,
            credentialDurationSeconds=3600
        )
    except iot.exceptions.ResourceAlreadyExistsException as e:
      # 同じ名前のPolicyがすでにある
      print(f"{tes_role_alias} already exists")
    print(tes_role_alias)

    echo_setup_command(init_info.region, init_info.device_name, tes_role, tes_role_alias)

if __name__ == "__main__":

    setup_device()