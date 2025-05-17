import requests
import json
from datetime import datetime

BASE_URL = "http://localhost:8000"


def print_response(response):
    print(f"\nStatus Code: {response.status_code}")
    try:
        print(json.dumps(response.json(), indent=2))
    except:
        print(response.text)
    print("-" * 80)


def test_list_videos():
    print("\n=== Testing List Videos Endpoint ===")

    # Test 1: Default parameters
    print("\nTest 1: Default parameters")
    response = requests.get(f"{BASE_URL}/videos")
    print_response(response)

    # Test 2: With pagination
    print("\nTest 2: With pagination (limit=2, offset=0)")
    response = requests.get(f"{BASE_URL}/videos?limit=2&offset=0")
    print_response(response)

    # Test 3: With has_alert filter
    print("\nTest 3: With has_alert filter")
    response = requests.get(f"{BASE_URL}/videos?has_alert=true")
    print_response(response)

    # Test 4: Invalid limit
    print("\nTest 4: Invalid limit")
    response = requests.get(f"{BASE_URL}/videos?limit=101")
    print_response(response)


def test_get_video():
    print("\n=== Testing Get Video Endpoint ===")

    # First get a list of videos to get a valid video_id
    response = requests.get(f"{BASE_URL}/videos?limit=1")
    if response.status_code == 200:
        videos = response.json()
        if videos:
            video_id = videos[0]["video_id"]

            # Test 1: Valid video_id
            print(f"\nTest 1: Get video with ID: {video_id}")
            response = requests.get(f"{BASE_URL}/videos/{video_id}")
            print_response(response)

    # Test 2: Invalid video_id
    print("\nTest 2: Invalid video_id")
    response = requests.get(f"{BASE_URL}/videos/nonexistent_id")
    print_response(response)


if __name__ == "__main__":
    print("Starting API Tests...")
    test_list_videos()
    test_get_video()
    print("\nAPI Tests Completed!")
