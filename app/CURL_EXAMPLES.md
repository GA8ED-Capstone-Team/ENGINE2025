# API Integration with cURL

Replace `<your-server-ip>` with your actual server IP address in all examples.

## 1. Health Check
```bash
# Check if API is running
curl http://<your-server-ip>:8000/api/v1/ping
```

Expected response:
```json
{
    "status": "ok",
    "message": "pong",
    "timestamp": "2024-03-21T10:30:00Z"
}
```

## 2. List Videos

### Basic List (Default Parameters)
```bash
# Get first 10 videos
curl http://<your-server-ip>:8000/api/v1/videos
```

### With Pagination
```bash
# Get 5 videos starting from offset 10
curl http://<your-server-ip>:8000/api/v1/videos?limit=5&offset=10
```

### With Alert Filter
```bash
# Get videos with alerts
curl http://<your-server-ip>:8000/api/v1/videos?has_alert=true

# Get videos without alerts
curl http://<your-server-ip>:8000/api/v1/videos?has_alert=false
```

### Combined Parameters
```bash
# Get 5 videos with alerts, starting from offset 10
curl http://<your-server-ip>:8000/api/v1/videos?limit=5&offset=10&has_alert=true
```

Expected response format:
```json
[
    {
        "video_id": "string",
        "video_uri": "string",
        "tracked_predictions_uri": "string",
        "annotated_video_uri": "string",
        "stability_score": 0.95,
        "bear_alert": true,
        "vandalism_genai_response": "string",
        "vandalism_alert": false,
        "created_at": "2024-03-21T10:30:00Z",
        "updated_at": "2024-03-21T10:30:00Z"
    }
]
```

## 3. Get Video Details

### Get Specific Video
```bash
# Replace video123 with actual video_id
curl http://<your-server-ip>:8000/api/v1/videos/video123
```

Expected response:
```json
{
    "video_id": "video123",
    "video_uri": "string",
    "tracked_predictions_uri": "string",
    "annotated_video_uri": "string",
    "stability_score": 0.95,
    "bear_alert": true,
    "vandalism_genai_response": "string",
    "vandalism_alert": false,
    "created_at": "2024-03-21T10:30:00Z",
    "updated_at": "2024-03-21T10:30:00Z"
}
```

## 4. Error Examples

### Invalid Video ID
```bash
curl http://<your-server-ip>:8000/api/v1/videos/nonexistent_id
```
Expected response (404):
```json
{
    "detail": "Video not found"
}
```

### Invalid Limit Parameter
```bash
curl http://<your-server-ip>:8000/api/v1/videos?limit=101
```
Expected response (422):
```json
{
    "detail": [
        {
            "loc": ["query", "limit"],
            "msg": "ensure this value is less than or equal to 100",
            "type": "value_error.number.not_le",
            "ctx": {"limit_value": 100}
        }
    ]
}
```

## 5. Pretty Printing

For better readability, you can use the `-s` (silent) and `-H` (header) options with `jq`:

```bash
# Install jq if not already installed
# On Ubuntu/Debian:
sudo apt-get install jq
# On macOS:
brew install jq

# Pretty print JSON response
curl -s http://<your-server-ip>:8000/api/v1/videos | jq '.'

# Pretty print specific video
curl -s http://<your-server-ip>:8000/api/v1/videos/video123 | jq '.'
```

## 6. Testing Script

You can also use the provided Python test script:
```bash
# Update BASE_URL in test_api.py first
python test_api.py
```

## 7. Common Issues

1. **Connection Refused**
   ```bash
   # Check if server is running
   curl -v http://<your-server-ip>:8000/api/v1/ping
   ```

2. **CORS Issues**
   ```bash
   # Test with CORS headers
   curl -H "Origin: http://localhost:3000" \
        -H "Access-Control-Request-Method: GET" \
        http://<your-server-ip>:8000/api/v1/videos
   ```

3. **Timeout Issues**
   ```bash
   # Set timeout
   curl --max-time 5 http://<your-server-ip>:8000/api/v1/videos
   ```

## 8. Environment Variables

For easier testing, set up environment variables:
```bash
# Set base URL
export API_BASE_URL="http://<your-server-ip>:8000/api/v1"

# Use in commands
curl $API_BASE_URL/videos
curl $API_BASE_URL/videos/video123
```

## 9. One-liners for Common Tasks

```bash
# Get total number of videos
curl -s $API_BASE_URL/videos | jq 'length'

# Get all video IDs
curl -s $API_BASE_URL/videos | jq '.[].video_id'

# Get videos with bear alerts
curl -s $API_BASE_URL/videos | jq '.[] | select(.bear_alert == true)'

# Get videos with vandalism alerts
curl -s $API_BASE_URL/videos | jq '.[] | select(.vandalism_alert == true)'

# Get videos sorted by stability score
curl -s $API_BASE_URL/videos | jq 'sort_by(.stability_score) | reverse'
``` 