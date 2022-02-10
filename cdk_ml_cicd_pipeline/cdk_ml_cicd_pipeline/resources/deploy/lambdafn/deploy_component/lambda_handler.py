import json
import boto3
from datetime import datetime as dt
import os

ggv2 = boto3.client('greengrassv2')
iot = boto3.client('iot')
dynamodb = boto3.resource("dynamodb")
table = dynamodb.Table(os.getenv("TABLE_NAME"))

def create_deployment(groupArn, deployment_name, component):
    """コンポーネントをデプロイする

    Attributes:
        groupArn (str):
            デプロイ対象のThingGroup ARN
        deployment_name (str):
            デプロイメントグループの名前
        component (str):
            デプロイするコンポーネントの情報

    Returns:
        dict: inputのdictの"status"をビルド結果でアップデートした値
    """
    response = ggv2.create_deployment(targetArn = groupArn, deploymentName = deployment_name, components = component)
    return response
    
def handler(event, context):
    """コンポーネントをデプロイする

    Attributes:
        event (dict):
            event data: {
                "component_name": "com.example.ggmlcomponent",
                "version": "1.0.0",
                "s3object": "s3://ml-model-build-input-us-east-1/newmodel.tar.gz",
                "group_name": "gggroup",
                "build_id": "ComponentInageBuild:f2d8dd48-97de-4a47-b280-aaaa34747c01",
                "status": "image_exists"
            }

    Returns:
        dict: inputのdictの"status"をビルド結果でアップデートした値
    """
    print(json.dumps(event))

    ### create component info
    components={
        event["component_name"]: {
            'componentVersion': event["version"],
        }
    }
    
    ### get deploy group arn
    deploy_tg = iot.describe_thing_group(thingGroupName = event['group_name'])
    
    ### create deployment name
    time = dt.now()
    deployment_name = 'auto-ml-cicd' + event['group_name']
    
    response = create_deployment(deploy_tg['thingGroupArn'], deployment_name, components)
    print("---deployment responese---\n{}".format(response))
    event['deploy_id'] = response['deploymentId']
    
    ### deployment status
    deploy_status = ggv2.get_deployment(deploymentId = response['deploymentId'])
    
    response = table.update_item(
        Key={
            "component_name": event["component_name"],
            "version": event["version"]
        },
        UpdateExpression="set deployment_status = :s, deploy_group = :g, update_time = :t",
        ExpressionAttributeValues={
            ":s": deploy_status["deploymentStatus"],
            ":g": deploy_tg['thingGroupArn'],
            ":t": dt.now().strftime('%Y-%m-%d %H:%M:%S')
        },
        ReturnValues="UPDATED_NEW"
    )
    
    event["status"] = "create_deployment"
    print(event)
    return event