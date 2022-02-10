import json
import boto3
from datetime import datetime as dt
import os

ggv2 = boto3.client('greengrassv2')
iot = boto3.client('iot')
dynamodb = boto3.resource("dynamodb")
table = dynamodb.Table(os.getenv("TABLE_NAME"))

class DeployFaild(Exception):
    """デプロイに失敗
    """
    def __init__(self, message):
        self.message = message

def check_deploy_job_status(job_id, thing_list):
    """コンポーネントをデプロイ状況を確認する

    Greengrassのデプロイは、Jobの結果を確認する必要があるので、対象となるデバイス全てに対してJobの状況をチェックする

    Attributes:
        job_id (str):
            デプロイ対象のThingGroup ARN
        thing_list (list):
            デプロイ対象のThingのリスト

    Returns:
        bool: True:デプロイ完了, False:デプロイ中
    """

    ### greengrass deployment
    gg_deploy = ggv2.get_deployment(deploymentId = job_id)

    deploy_failed = False
    
    for thing in thing_list:
        response = iot.describe_job_execution(
            jobId=gg_deploy["iotJobId"],
            thingName=thing
        )
        print(response)
        if response["execution"]["status"] in ["QUEUED", "IN_PROGRESS"]:
            print("{} is deploying. status: {}".format(thing, response["execution"]["status"]))
            return False
        elif response["execution"]["status"] == "FAILED":
            raise DeployFaild(thing + " faild to deploy. reson: " + json.dumps(response["execution"]["statusDetails"]))

    return True

def get_thing_list(group_name):
    """Thing Groupに含まれているThingを取得する

    Groupに登録されているThingが多い場合は、next tokenを使って全件取得するようにしてください

    Attributes:
        group_name (str):
            Thing Group名

    Returns:
        list: Thing nameのリスト
    """

    response = iot.list_things_in_thing_group(
        thingGroupName=group_name
    )
    
    return response["things"]

def handler(event, context):
    """デプロイ結果を確認する

    Attributes:
        event (dict):
            event data: {
                "component_name": "com.example.ggmlcomponent",
                "version": "1.0.0",
                "s3object": "s3://ml-model-build-input-us-east-1/newmodel.tar.gz",
                "group_name": "MLOps_things_group",
                "status": "create_deployment",
                "build_id": "",
                "deploy_id": "32a8ea51-2405-4137-bd31-9b4481ed7945"
            }

    Returns:
        dict: inputのdictの"status"をビルド結果でアップデートした値
    """
    print(json.dumps(event))

    ### get deploy group arn
    deploy_tg = iot.describe_thing_group(thingGroupName = event['group_name'])
    
    ### get thing list
    things = get_thing_list(event['group_name'])
    
    ### deployment status
    try:
        if check_deploy_job_status(event['deploy_id'], things):
            event["status"] = "COMPLETED"
            event["message"] = "Deployment is finished."
        else:
            event["status"] = "RUNNING"
    except DeployFaild as e:
        event["status"] = "FAILED"
        event["message"] = e.message
        print(e)

    print(event["status"])

    response = table.update_item(
        Key={
            "component_name": event["component_name"],
            "version": event["version"]
        },
        UpdateExpression="set deployment_status = :s, deploy_group = :g, update_time = :t",
        ExpressionAttributeValues={
            ":s": event["status"],
            ":g": deploy_tg['thingGroupArn'],
            ":t": dt.now().strftime('%Y-%m-%d %H:%M:%S')
        },
        ReturnValues="UPDATED_NEW"
    )
   
    print(event)
    return event