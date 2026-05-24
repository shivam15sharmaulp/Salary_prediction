from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field
from sklearn.model_selection import train_test_split

from starter.ml.data import process_data
from starter.ml.model import inference

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "census.csv"
MODEL_PATH = BASE_DIR / "model" / "model.joblib"
LABEL_COLUMN = "salary"
CATEGORICAL_FEATURES = [
	"workclass",
	"education",
	"marital-status",
	"occupation",
	"relationship",
	"race",
	"sex",
	"native-country",
]


class CensusRecord(BaseModel):
	age: int
	workclass: str
	fnlgt: int
	education: str
	education_num: int = Field(..., alias="education-num")
	marital_status: str = Field(..., alias="marital-status")
	occupation: str
	relationship: str
	race: str
	sex: str
	capital_gain: int = Field(..., alias="capital-gain")
	capital_loss: int = Field(..., alias="capital-loss")
	hours_per_week: int = Field(..., alias="hours-per-week")
	native_country: str = Field(..., alias="native-country")

	model_config = ConfigDict(
		populate_by_name=True,
		json_schema_extra={
			"example": {
				"age": 37,
				"workclass": "Private",
				"fnlgt": 34146,
				"education": "Bachelors",
				"education-num": 13,
				"marital-status": "Married-civ-spouse",
				"occupation": "Exec-managerial",
				"relationship": "Husband",
				"race": "White",
				"sex": "Male",
				"capital-gain": 0,
				"capital-loss": 0,
				"hours-per-week": 50,
				"native-country": "United-States",
			}
		},
	)


class PredictionResponse(BaseModel):
	prediction: str


app = FastAPI(title="Census Income Prediction API")


@lru_cache(maxsize=1)
def load_artifacts() -> dict[str, object]:
	data = pd.read_csv(DATA_PATH)
	data.columns = data.columns.str.strip()

	train, _ = train_test_split(data, test_size=0.20, random_state=42)
	_, _, encoder, lb = process_data(
		train,
		categorical_features=CATEGORICAL_FEATURES,
		label=LABEL_COLUMN,
		training=True,
	)

	return {
		"model": joblib.load(MODEL_PATH),
		"encoder": encoder,
		"lb": lb,
	}


def normalize_features(record: CensusRecord) -> pd.DataFrame:
	features = pd.DataFrame([record.model_dump(by_alias=True)])

	for feature in CATEGORICAL_FEATURES:
		features[feature] = features[feature].map(lambda value: f" {str(value).strip()}")

	return features


@app.get("/")
def welcome() -> dict[str, str]:
	return {"message": "Welcome to the census income prediction API."}


@app.post("/predict", response_model=PredictionResponse)
def predict_salary(record: CensusRecord) -> PredictionResponse:
	artifacts = load_artifacts()
	features = normalize_features(record)

	processed_features, _, _, _ = process_data(
		features,
		categorical_features=CATEGORICAL_FEATURES,
		training=False,
		encoder=artifacts["encoder"],
		lb=artifacts["lb"],
	)
	prediction = inference(artifacts["model"], processed_features)
	label = str(artifacts["lb"].classes_[int(prediction[0])]).strip()

	return PredictionResponse(prediction=label)


if __name__ == "__main__":
	import uvicorn

	uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
