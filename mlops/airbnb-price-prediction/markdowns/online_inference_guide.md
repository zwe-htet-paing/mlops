# Sample online inference

Use the following CURL command to get real-time predictions:


```curl
curl --location 'http://localhost:6789/api/runs' \
--header 'Authorization: Bearer bb22ef7fdc1346d38058a44206832eb1' \
--header 'Content-Type: application/json' \
--header 'Cookie: lng=en' \
--data '{
    "run": {
        "pipeline_uuid": "predict",
        "block_uuid": "inference",
        "variables": {
            "inputs": [
                {
                    "room_type": "Private room",
                    "neighbourhood": "FOURTEENTH WARD",
                    "latitude": 42.66719,
                    "longitude": -73.8158,
                    "minimum_nights": 1,
                    "number_of_reviews": 250,
                    "reviews_per_month": 1.93,
                    "availability_365": 255
                },
                {
                    "room_type": "Private room",
                    "neighbourhood": "FOURTEENTH WARD",
                    "latitude": 42.66719,
                    "longitude": -73.8158,
                    "minimum_nights": 1,
                    "number_of_reviews": 250,
                    "reviews_per_month": 1.93,
                    "availability_365": 255
                }
            ]
        }
    }
}'

```

## Note

The `Authorization` header is using this pipeline’s API trigger’s token value.
The token value is set to `fire` for this project.
If you create a new trigger, that token will change.
Only use a fixed token for testing or demonstration purposes.