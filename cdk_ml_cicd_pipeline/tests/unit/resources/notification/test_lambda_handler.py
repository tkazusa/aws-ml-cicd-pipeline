from cdk_ml_cicd_pipeline.resources.notification.lambdafn.lambda_handler import extract_consolelink_url


def test_send_slack_message():
    """
    Amazon SNS からメッセージを受け取った場合に Lambda handler が受け取る event から、Slack に送信する CodePipeline Manual Approval へのコンソールリンクを抽出するテスト。
    SNS から受け取る Message の値が str な点に注意。
    """
    event = {
        "Records": [
            {
                "EventSource": "aws:sns",
                "EventVersion": "1.0",
                "EventSubscriptionArn": "arn:aws:sns:region:account-id:Slack-notify:resource-id",
                "Sns": {
                    "Type": "Notification",
                    "MessageId": "43eafeacb-e3af-5sdfe-9b45-e0fdsdfsaew",
                    "TopicArn": "arn:aws:sns:region:account-id:Slack-notify",
                    "Subject": "APPROVAL NEEDED: AWS CodePipeline ml-cicd-pipelin... for action approval",
                    "Message": '{"region":"us-east-1","consoleLink":"https://console.aws.amazon.com/codesuite/codepipeline/pipelines/ml-cicd-pipeline/view?region=us-east-1","approval":{"pipelineName":"taketosk-ml-cicd-pipeline","stageName":"approval","actionName":"approval","token":"50fd00f7-d5e6-4a6c-a9cb-69a72338ce52","expires":"2021-11-18T02:45Z","externalEntityLink":null,"approvalReviewLink":"https://console.aws.amazon.com/codesuite/codepipeline/pipelines/taketosk-ml-cicd-pipeline/view?region=us-east-1#/approval/approval/approve/50fd00f7-d5e6-4a6c-a9cb-69a72338ce52","customData":null}}',
                    "Timestamp": "2021-XX-XXT00:00:00.000Z",
                    "SignatureVersion": "1",
                    "Signature": "sdfaeEAdfaeFAEveFAEFA",
                    "SigningCertUrl": "https://sns.us-east-0.amazonaws.com/SimpleNotificationService-7ff5318490ec183fbaddaa2a969abfda.pem",
                    "UnsubscribeUrl": "https://sns.us-east-0.amazonaws.com/?Action=Unsubscribe&SubscriptionArn=arn:aws:sns:region:account-id:Slack-notify:resource-id",
                    "MessageAttributes": {},
                },
            }
        ]
    }

    expected = "https://console.aws.amazon.com/codesuite/codepipeline/pipelines/ml-cicd-pipeline/view?region=us-east-1"
    acctual = extract_consolelink_url(event)
    expected = acctual
    assert expected == acctual
