import requests
import json
from datetime import datetime

BASE_URL = "http://localhost:8000/api/v1"


def print_response(response):
    print(f"\nStatus Code: {response.status_code}")
    try:
        print(json.dumps(response.json(), indent=2))
    except:
        print(response.text)
    print("-" * 80)


def assert_response(response, expected_status=200):
    if response.status_code != expected_status:
        raise AssertionError(
            f"Expected status code {expected_status}, got {response.status_code}. Response: {response.text}"
        )
    return response.json()


def test_list_videos():
    print("\n=== Testing List Videos Endpoint ===")

    # Test 1: Default parameters
    print("\nTest 1: Default parameters")
    response = requests.get(f"{BASE_URL}/videos")
    data = assert_response(response)
    assert isinstance(data, list), "Response should be a list"
    if data:
        assert all(
            isinstance(item, dict) for item in data
        ), "All items should be dictionaries"
        assert all(
            "video_id" in item for item in data
        ), "All items should have video_id"

    # Test 2: With pagination
    print("\nTest 2: With pagination (limit=2, offset=0)")
    response = requests.get(f"{BASE_URL}/videos?limit=2&offset=0")
    data = assert_response(response)
    assert len(data) <= 2, "Should return at most 2 items"

    # Test 3: With has_alert filter
    print("\nTest 3: With has_alert filter")
    response = requests.get(f"{BASE_URL}/videos?has_alert=true")
    data = assert_response(response)
    if data:
        assert all(
            item.get("bear_alert") or item.get("vandalism_alert") for item in data
        ), "All returned items should have either bear_alert or vandalism_alert"

    # Test 4: Invalid limit
    print("\nTest 4: Invalid limit")
    response = requests.get(f"{BASE_URL}/videos?limit=101")
    assert_response(response, expected_status=422)  # Validation error


def test_get_video():
    print("\n=== Testing Get Video Endpoint ===")

    # First get a list of videos to get a valid video_id
    response = requests.get(f"{BASE_URL}/videos?limit=1")
    data = assert_response(response)

    if data:
        video_id = data[0]["video_id"]

        # Test 1: Valid video_id
        print(f"\nTest 1: Get video with ID: {video_id}")
        response = requests.get(f"{BASE_URL}/videos/{video_id}")
        data = assert_response(response)
        assert isinstance(data, dict), "Response should be a dictionary"
        assert (
            data["video_id"] == video_id
        ), "Returned video_id should match requested id"
        assert all(
            key in data
            for key in [
                "video_uri",
                "tracked_predictions_uri",
                "annotated_video_uri",
                "stability_score",
                "bear_alert",
                "vandalism_genai_response",
                "vandalism_alert",
                "created_at",
                "updated_at",
            ]
        ), "Response should contain all required fields"

    # Test 2: Invalid video_id
    print("\nTest 2: Invalid video_id")
    response = requests.get(f"{BASE_URL}/videos/nonexistent_id")
    assert_response(response, expected_status=404)  # Not found error


if __name__ == "__main__":
    print("Starting API Tests...")
    try:
        test_list_videos()
        test_get_video()
        print("\n✅ All tests passed successfully!")
    except AssertionError as e:
        print(f"\n❌ Test failed: {str(e)}")
        raise
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        raise
