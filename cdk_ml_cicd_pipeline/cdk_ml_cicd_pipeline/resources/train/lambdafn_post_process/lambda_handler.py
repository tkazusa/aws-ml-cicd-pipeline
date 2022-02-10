import os
import json
from typing import Any, Dict
import boto3

import requests

lambda_client = boto3.client('lambda')
codepipeline_client = boto3.client('codepipeline')

# Slack の Webhook のための URL
webhookurl = os.environ['slack_hook_url']


def send_slack_message(message: str) -> requests.Response:
    """Send a meesage to slack webhook url"""
    headers = {"Content-Type": "application/json"}
    response = requests.post(url=webhookurl, headers=headers, json={"text": message})
    return response


def lambda_handler(event, context) -> Dict[str, Any]:
    try:
        ###
        # Get CodepipeLine Job ID
        ###
        
        job_id = event['CodePipeline.job']['id']
        job_data = event['CodePipeline.job']['data']
        user_params = json.loads(job_data['actionConfiguration']['configuration']['UserParameters'])

        ###
        # Process
        ###
        
        experiment_name = user_params["EXPERIMENT_NAME"]
        run_id = user_params["RUN_ID"]
        train_job_name = user_params['TRAIN_JOB_NAME']
        trained_model_s3 = user_params['TRAINED_MODEL_S3']

        experiment_info = f"Experiment name: {experiment_name}\n" + f"Run ID: {run_id}\n"
        model_info = f"TRAIN_JOB_NAME: {train_job_name}\n" + f"TRAINED_MODEL_S3: {trained_model_s3}\n"
        
        message = "".join(["手動承認が完了しました。\n", experiment_info, model_info])
        response = send_slack_message(message)

        ### 
        # Response job result(success) to codepipeline
        ###
        
        codepipeline_client.put_job_success_result(jobId=job_id)  

        return {"statusCode": 200, "body": message}

    except Exception as e:
        ### 
        # Response job result(failure) to codepipeline
        ###

        codepipeline_client.put_job_failure_result(
            jobId=job_id,
            failureDetails={
                'type': 'JobFailed',
                'message': str(e)
            }
        ) 
        
        raise Exception(e)

