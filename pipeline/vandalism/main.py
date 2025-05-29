import os
from utils import (
    get_video_id_from_s3_path,
    update_video_record,
    contains_person_or_car,
    get_video_path_from_db,
    download_video_from_s3,
    get_gemini_api_key,
    analyze_video_with_gemini,
)


def main():
    tracked_predictions = os.getenv("TRACKED_PREDICTIONS", "")
    video_id = get_video_id_from_s3_path(tracked_predictions)

    # Custom logic to only prompt VLM if there are person or car detections
    if not contains_person_or_car(tracked_predictions):
        print("No person or car detections found. Skipping Gemini API call.")
        # Update database with no vandalism
        update_video_record(video_id, "No person or car detected", False)
        return

    video_s3_uri = get_video_path_from_db(video_id)
    video_path = download_video_from_s3(video_s3_uri)
    gemini_api_key = get_gemini_api_key()
    vandalism_alert, vandalism_response = analyze_video_with_gemini(
        video_path, gemini_api_key
    )
    # Update database with vandalism analysis results
    print(f"Updating video record {video_id} for vandalism")
    update_video_record(video_id, vandalism_response, vandalism_alert)


if __name__ == "__main__":
    main()
