import os
import json
import urllib
from typing import Any, Dict
import boto3

codepipeline_client = boto3.client('codepipeline')
s3_client = boto3.client('s3')


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
        
        eval_report_s3 = user_params['EVAL_REPORT_S3']
        o = urllib.parse.urlparse(eval_report_s3)
        s3_bucket = o.hostname
        s3_object = o.path[1:]

        response = s3_client.get_object(Bucket=s3_bucket, Key=s3_object)
        body = response["Body"].read()

        experiment_info = json.loads(body.decode('utf-8'))
        experiment_info_str = {key: str(val) for key, val in experiment_info.items()}
        
        # `experiment_info` example:
        # {"average_precision": 0.7987789365647856, "experiment_name": "mlops-demo", "experiment_id": "2", "run_id": "541e1a56097a44659e0d588fb3cb51ec"}
        
        
        ### 
        # Response job result(success) to codepipeline
        ###
        
        codepipeline_client.put_job_success_result(
            jobId=job_id,
            outputVariables=experiment_info_str,
        )  

        return {"statusCode": 200, "body": body}

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

