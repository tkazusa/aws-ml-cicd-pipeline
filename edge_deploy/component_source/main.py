import time
import datetime
import json
import inference
import awsiot.greengrasscoreipc
from awsiot.greengrasscoreipc.model import (
    PublishToTopicRequest,
    PublishMessage,
    JsonMessage
)

# from edge_deploy.component_source.src.inference import Inference

# TIMEOUT = 10
# ipc_client = awsiot.greengrasscoreipc.connect()

inference()
                    
# topic = "mlops/inference/result"

# while True:
#     message = {
#       "timestamp": str(datetime.datetime.now()),
#       "value":mlmodel.inference()
#     }
#     message_json = json.dumps(message).encode('utf-8')

#     request = PublishToTopicRequest()
#     request.topic = topic
#     publish_message = PublishMessage()
#     publish_message.json_message = JsonMessage()
#     publish_message.json_message.message = message
#     request.publish_message = publish_message
#     operation = ipc_client.new_publish_to_topic()
#     operation.activate(request)
#     future = operation.get_response()
#     future.result(TIMEOUT)
    
    
#     print("publish")
#     time.sleep(1)