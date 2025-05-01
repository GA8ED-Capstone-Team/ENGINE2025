import os
import json
import boto3
import tempfile
from urllib.parse import urlparse

from utils import compute_stability_scores

WILD_ANIMALS = {
    21: "bear",
}
threshold = float(os.environ.get("STABILITY_THRESHOLD", "0.5"))
s3 = boto3.client("s3")


def download_predictions_from_s3(s3_uri):
    parsed = urlparse(s3_uri)
    bucket, key = parsed.netloc, parsed.path.lstrip("/")
    local_path = os.path.join(tempfile.gettempdir(), os.path.basename(key))
    s3.download_file(bucket, key, local_path)
    return local_path, bucket, key


def upload_json_to_s3(json_data, bucket, prefix, filename="stability_scores.json"):
    output_path = os.path.join(tempfile.gettempdir(), filename)
    with open(output_path, "w") as f:
        json.dump(json_data, f, indent=2)
    s3.upload_file(output_path, bucket, f"{prefix}/{filename}")
    print(f"✅ Uploaded results to s3://{bucket}/{prefix}/{filename}")


def main():

    input_s3 = os.environ["TRACKED_PREDICTIONS"]
    local_predictions, bucket, key = download_predictions_from_s3(input_s3)

    # Compute stability scores
    local_stability_scores = os.path.join(
        tempfile.gettempdir(), "stability_scores.json"
    )
    stability_scores = compute_stability_scores(
        json_input_path=local_predictions,
        json_output_path=local_stability_scores,
        stability_threshold=threshold,
        wild_animals=WILD_ANIMALS,
        raise_alerts=True,
    )

    output_prefix = f"object_detection_results/{os.path.dirname(key).split('/')[-1]}"
    print(key)
    print("Uploading stability scores to S3...")
    upload_json_to_s3(stability_scores, bucket, output_prefix)


if __name__ == "__main__":
    main()
