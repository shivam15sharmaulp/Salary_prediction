import argparse
import json

import requests


DEFAULT_PAYLOAD = {
    "age": 52,
    "workclass": "Self-emp-not-inc",
    "fnlgt": 209642,
    "education": "HS-grad",
    "education-num": 9,
    "marital-status": "Married-civ-spouse",
    "occupation": "Exec-managerial",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital-gain": 0,
    "capital-loss": 0,
    "hours-per-week": 45,
    "native-country": "United-States",
}


def post_inference(url: str) -> tuple[int, str]:
    response = requests.post(url, json=DEFAULT_PAYLOAD, timeout=30)
    response.raise_for_status()

    prediction = response.json()["prediction"]
    return response.status_code, prediction


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Send a prediction request to the deployed FastAPI app."
    )
    parser.add_argument(
        "--url",
        default="http://127.0.0.1:8000/predict",
        help="Prediction endpoint URL.",
    )
    args = parser.parse_args()

    try:
        status_code, prediction = post_inference(args.url)
    except requests.RequestException as exc:
        print(json.dumps({"error": str(exc), "url": args.url}, indent=2))
        raise SystemExit(1) from exc

    print(
        json.dumps(
            {
                "status_code": status_code,
                "prediction": prediction,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()