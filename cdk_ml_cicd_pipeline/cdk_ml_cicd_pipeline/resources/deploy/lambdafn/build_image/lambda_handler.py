import boto3
from datetime import datetime as dt
import json
import os

dynamodb = boto3.resource("dynamodb")
table = dynamodb.Table(os.getenv("TABLE_NAME"))
codebuild_project_name = os.getenv("CODEBUILD_PROJECT_NAME")
component_image_base_repo = os.getenv("COMPONENT_BASE_IMAGE_REPOSITORY")
component_image_repo = os.getenv("COMPONENT_IMAGE_REPOSITORY")
component_app_repo = os.getenv("COMPONENT_APP_SOURCE_REPOSITORY")

def handler(event, context):
    """コンポーネントで利用するコンテナイメージを作成する

    Attributes:
        event (dict):
            event data: {
                "component_name": "com.example.ggmlcomponent",
                "version": "1.0.0",
                "s3object": "s3://ml-model-build-input-us-east-1/newmodel.tar.gz",
                "group_name": "gggroup"
            }

    Returns:
        dict: inputのdictに"status"を追加した値。
    """

    # デプロイしようとしているモデルの情報があるかを確認
    response = table.get_item(
      Key={
          "component_name": event["component_name"],
          "version": event["version"]
      }
    )
    print(response)
    
    if "Item" not in response or "component_arn" not in  response["Item"]:
        # エントリー自体または、コンポーネントが存在しないので、ビルド処理をすすめる
        event["status"] = "image_creating"
    else:
        # コンポーネントが存在するので、デプロイに飛ぶようにステータスを指定
        event["status"] = "component_exists"
        event["build_id"] = ""
        return event

    # DDBにエントリーを追加
    item = {
        "component_name": event["component_name"],
        "version": event["version"],
        "pipeline_status": event["status"],
        "bucket": event["s3object"].split("/")[2],
        "s3_path": "/".join(event["s3object"].split("/")[3:]),
        "update_time": dt.now().strftime('%Y-%m-%d %H:%M:%S'),
    }
    table.put_item(
        Item=item
    )

    # CodeBuildを使ってdockerイメージをビルド
    environment_variable = [
        {
            "name": "COMPONENT_IMAGE_REPOSITORY",
            "value": component_image_repo,
            "type": "PLAINTEXT"
        },
        {
            "name": "COMPONENT_BASE_IMAGE_REPOSITORY",
            "value": component_image_base_repo,
            "type": "PLAINTEXT"
        },
        {
            "name": "IMAGE_TAG",
            "value": event["version"],
            "type": "PLAINTEXT"
        },
        {
            "name": "S3_URI",
            "value": event["s3object"],
            "type": "PLAINTEXT"
        },
        {
            "name": "FILE_NAME",
            "value": event["s3object"].split("/")[-1],
            "type": "PLAINTEXT"
        },
        {
            "name": "COMPONENT_APP_SOURCE_REPOSITORY",
            "value": component_app_repo,
            "type": "PLAINTEXT"
        },
        {
            "name": "ACCOUNT_ID",
            "value": boto3.client("sts").get_caller_identity()["Account"],
            "type": "PLAINTEXT"
        }
    ]

    client = boto3.client('codebuild')
    res = client.start_build(projectName=codebuild_project_name,
        environmentVariablesOverride=environment_variable)

    event["build_id"] = res["build"]["id"]
    print(event)

    return event