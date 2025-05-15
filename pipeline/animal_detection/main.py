import os
import tempfile
from utils import (
    compute_stability_scores,
    get_video_id_from_s3_path,
    update_video_record,
    upload_json_to_s3,
    download_predictions_from_s3,
    WILD_ANIMALS,
)


def main():
    input_s3 = os.environ["TRACKED_PREDICTIONS"]
    local_predictions, bucket, key = download_predictions_from_s3(input_s3)
    video_id = get_video_id_from_s3_path(input_s3)

    # Compute stability scores
    local_stability_scores = os.path.join(
        tempfile.gettempdir(), "stability_scores.json"
    )
    animal_alerts, stability_scores = compute_stability_scores(
        json_input_path=local_predictions,
        json_output_path=local_stability_scores,
        raise_alerts=True,
    )

    # Get stability scores for all wild animals
    animal_scores = {}
    for class_id, animal_name in WILD_ANIMALS.items():
        score = stability_scores.get(class_id, {}).get("stability_score", None)
        animal_scores[animal_name] = score

    # This aggrergation logic can be tweaked if needed
    max_stability_score = round(
        max(
            (score for score in animal_scores.values() if score is not None),
            default=0,
        ),
        4,
    )

    # Custom logic: Any wild animal alert triggers the bear_alert (can be refined)
    any_wild_animal_alert = any(animal_alerts.values())
    update_video_record(video_id, max_stability_score, any_wild_animal_alert)
    output_prefix = f"object_detection_results/{video_id}"
    print("Uploading stability scores to S3")
    upload_json_to_s3(
        stability_scores, bucket, output_prefix, filename="stability_scores.json"
    )


if __name__ == "__main__":
    main()
