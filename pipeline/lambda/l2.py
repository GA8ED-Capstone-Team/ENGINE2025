import json
import boto3
import os
import urllib.parse
from typing import Dict, List

ecs = boto3.client("ecs")

# === CONFIG ===
CLUSTER = "ga8ed-ecs-cluster"
SUBNETS = [os.environ["L2_SUBNET_ID"]]
SECURITY_GROUPS = [os.environ["L2_SECURITY_GROUP_ID"]]

# Task configurations
# TODO: Need to add speed estimation here
TASKS = [
    {
        "name": "vandalism",
        "task_definition": "vandalism",
    },
    {
        "name": "stab-score",
        "task_definition": "stab-score",
        "environment_variables": {
            "STABILITY_THRESHOLD": "0.2",
        },
    },
]


def run_ecs_task(
    task_config: Dict,
    tracked_predictions: str,
    cluster: str,
    subnets: List[str],
    security_groups: List[str],
) -> Dict:
    """
    Run an ECS task with the given configuration.

    Args:
        task_config: Dictionary containing task configuration
        tracked_predictions: S3 path to tracked predictions
        cluster: ECS cluster name
        subnets: List of subnet IDs
        security_groups: List of security group IDs

    Returns:
        Response from ECS run_task API
    """
    # Base environment variables that all tasks need
    env_vars = [{"name": "TRACKED_PREDICTIONS", "value": tracked_predictions}]

    # Add task-specific environment variables if any
    if "environment_variables" in task_config:
        for name, value in task_config["environment_variables"].items():
            env_vars.append({"name": name, "value": value})

    # Run the task
    response = ecs.run_task(
        cluster=cluster,
        launchType="FARGATE",
        taskDefinition=task_config["task_definition"],
        networkConfiguration={
            "awsvpcConfiguration": {
                "subnets": subnets,
                "securityGroups": security_groups,
                "assignPublicIp": "ENABLED",
            }
        },
        overrides=[{"name": task_config["name"], "environment": env_vars}],
    )

    print(
        f"{task_config['name']} ECS Task launched:",
        json.dumps(response, indent=4, default=str),
    )
    return response


def lambda_handler(event, context):
    print("Received event:", json.dumps(event, indent=2))

    # Extract the S3 tracked predictions path from event
    bucket = event["Records"][0]["s3"]["bucket"]["name"]
    key = urllib.parse.unquote_plus(event["Records"][0]["s3"]["object"]["key"])
    tracked_predictions = f"s3://{bucket}/{key}"
    print("Launching ECS tasks with tracked predictions:", tracked_predictions)

    # Run all configured tasks
    responses = []
    for task_config in TASKS:
        response = run_ecs_task(
            task_config=task_config,
            tracked_predictions=tracked_predictions,
            cluster=CLUSTER,
            subnets=SUBNETS,
            security_groups=SECURITY_GROUPS,
        )
        responses.append(response)

    return {"statusCode": 200, "body": json.dumps("L2 tasks triggered successfully.")}
