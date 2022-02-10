import boto3
from datetime import datetime as dt
import json
import os

dynamodb = boto3.resource("dynamodb")
table = dynamodb.Table(os.getenv("TABLE_NAME"))
component_app_repo = os.getenv("COMPONENT_APP_SOURCE_REPOSITORY")

def get_recipe(params, aws_account_id):
    """コンポーネント用のレシピファイルを取得する

    Attributes:
        params (dict):
            event data: {
                "component_name": "com.example.ggcomponent",
                "version": "1.0.0",
                "s3object": "s3://ml-model-build-input-us-east-1/newmodel.tar.gz",
                "group_name": "gggroup",
                "build_id": "ComponentInageBuild:f2d8dd48-97de-4a47-b280-aaaa34747c01",
                "status": "image_exists"
            }

    Returns:
        dict: コンポーネントのRecipeデータ
    """
    client = boto3.client('codecommit')
    response = client.get_file(
        repositoryName=component_app_repo.split("//")[1],
        filePath="recipe.yaml"
    )
    recipe_template = response["fileContent"].decode('utf-8')
    image_name = "{}.dkr.ecr.{}.amazonaws.com/{}:{}".format(
        aws_account_id,
        os.getenv("AWS_DEFAULT_REGION"),
        params["component_name"],
        params["version"])
    recipe = recipe_template.replace("__VERSION__", params["version"]).replace("__IMAGE__", image_name)
    print(recipe)

    return recipe

def create_component(recipe):
    """コンポーネントを作成する

    Attributes:
        recipe (dict):
            コンポーネントのRecipeデータ

    Returns:
        str: コンポーネントのバージョンARN
    """
    ggv2 = boto3.client('greengrassv2')
    response = ggv2.create_component_version(
        inlineRecipe=recipe.encode('utf-8'),
    )
    print(response)

    return response["arn"]

def handler(event, context):
    """コンポーネントを作成する

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
    aws_account_id = context.invoked_function_arn.split(":")[4]


    # 作成しようとしているコンポーネントがすでに存在するか確認
    response = table.get_item(
      Key={
          "component_name": event["component_name"],
          "version": event["version"]
      }
    )
    print(response)
    
    if "component_arn" not in response["Item"]:
        pass
    else:
        # すでに存在しているので、作成せずに抜ける
        event["pipeline_status"] = "component_exists"
        return event

    # コンポーネントのレシピを取得
    recipe = get_recipe(event, aws_account_id)

    # コンポーネントの作成
    component_arn = create_component(recipe)
    event["status"] = "component_exists"
    event["component_arn"] = component_arn

    # DynamoDBのステータスを更新
    response = table.update_item(
        Key={
            "component_name": event["component_name"],
            "version": event["version"]
        },
        UpdateExpression="set pipeline_status = :s, update_time = :t, component_arn = :c",
        ExpressionAttributeValues={
            ":s": event["status"],
            ":c": event["component_arn"],
            ":t": dt.now().strftime('%Y-%m-%d %H:%M:%S')
        },
        ReturnValues="UPDATED_NEW"
    )
    print(event)

    return event