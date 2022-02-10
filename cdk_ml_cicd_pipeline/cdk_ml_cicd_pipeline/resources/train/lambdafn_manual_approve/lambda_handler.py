import os
import json
from typing import Any, Dict

import requests

# Slack の Webhook のための URL
webhookurl = os.environ['slack_hook_url']


def send_slack_message(message: str) -> requests.Response:
    """Send a meesage to slack webhook url"""
    headers = {"Content-Type": "application/json"}
    response = requests.post(url=webhookurl, headers=headers, json={"text": message})
    return response


def extract_sns_message(event) -> dict:
    message_dict = json.loads(event["Records"][0]["Sns"]["Message"])
    return message_dict


def lambda_handler(event, context) -> Dict[str, Any]:
    sns_message = extract_sns_message(event)
    # console_link = sns_message["consoleLink"]
    approval_review_link = sns_message["approval"]["approvalReviewLink"]
    dashboard_link = sns_message["approval"]["externalEntityLink"]
    
    experiment_info = f'{sns_message["approval"]["customData"]}\n'
    review_info = f"Approval Review Link: {approval_review_link}\n" + f"MLFlow Link: {dashboard_link}"

    message = "".join(["学習が完了しました。\n", experiment_info, review_info])
    response = send_slack_message(message)

    return {"statusCode": 200, "body": message}
