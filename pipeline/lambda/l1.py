import json
import boto3
import os
import urllib.parse

ecs = boto3.client("ecs")

# === CONFIG ===
CLUSTER = "ga8ed-ecs-cluster"
TASK_DEFINITION = "L1_task"
CONTAINER_NAME = "video-detect"


SUBNETS = [os.environ["L1_SUBNET_ID"]]
SECURITY_GROUPS = [os.environ["L1_SECURITY_GROUP_ID"]]

def lambda_handler(event, context):

    print("Received event:", json.dumps(event, indent=2))

    # Extract the S3 video path from event
    bucket = event['Records'][0]['s3']['bucket']['name']
    key = urllib.parse.unquote_plus(event['Records'][0]['s3']['object']['key'])
    video_s3_path = f"s3://{bucket}/{key}"
    print("Launching ECS task with video:", video_s3_path)

    # Run the ECS Fargate task
    response = ecs.run_task(
        cluster=CLUSTER,
        launchType="FARGATE",
        taskDefinition=TASK_DEFINITION,
        networkConfiguration={
            "awsvpcConfiguration": {
                "subnets": SUBNETS,
                "securityGroups": SECURITY_GROUPS,
                "assignPublicIp": "ENABLED"
            }
        },
        overrides={
            "containerOverrides": [{
                "name": CONTAINER_NAME,
                "environment": [
                    {"name": "VIDEO_S3_PATH", "value": video_s3_path}
                ]
            }]
        }
    )
    print("ECS Task launched:", json.dumps(response, indent=4, default=str))

    return {
        "statusCode": 200,
        "body": json.dumps("L1_task triggered successfully.")
    }
