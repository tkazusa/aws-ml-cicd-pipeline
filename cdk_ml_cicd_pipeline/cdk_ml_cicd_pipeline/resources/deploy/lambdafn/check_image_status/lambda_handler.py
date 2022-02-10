import boto3
from datetime import datetime as dt
import json
import os

dynamodb = boto3.resource("dynamodb")
table = dynamodb.Table(os.getenv("TABLE_NAME"))
def handler(event, context):
    """コンポーネントで利用するコンテナイメージを作成する

    Attributes:
        event (dict):
            event data: {
                "component_name": "com.example.ggmlcomponent",
                "version": "1.0.0",
                "s3object": "s3://ml-model-build-input-us-east-1/newmodel.tar.gz",
                "group_name": "gggroup",
                "build_id": "ComponentInageBuild:f2d8dd48-97de-4a47-b280-aaaa34747c01",
                "status": "image_creating"
            }

    Returns:
        dict: inputのdictの"status"をビルド結果でアップデートした値
    """
    print(json.dumps(event))

    # CodeBuildのビルドステータスを取得
    client = boto3.client('codebuild')
    response = client.batch_get_builds(
        ids=[
            event["build_id"],
        ]
    )
    print(response)


    if response["builds"][0]["buildStatus"] == "IN_PROGRESS":
        return event
    elif response["builds"][0]["buildStatus"] == "SUCCEEDED":
        event["status"] = "image_exists"
    else:
        event["status"] = "image_faild"
        event["message"] = "Building component image faild."

    # DynamoDBのステータスを更新
    response = table.update_item(
        Key={
            "component_name": event["component_name"],
            "version": event["version"]
        },
        UpdateExpression="set pipeline_status = :s, update_time = :t",
        ExpressionAttributeValues={
            ":s": event["status"],
            ":t": dt.now().strftime('%Y-%m-%d %H:%M:%S')
        },
        ReturnValues="UPDATED_NEW"
    )
    print(event)

    return event