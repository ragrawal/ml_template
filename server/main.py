import os
from typing import Annotated

import joblib
from fastapi import Body, FastAPI
from pandera.typing import DataFrame

from ml_template.models.estimator import MyEstimator

model = joblib.load(os.environ["MODEL"])
app = FastAPI()


class PredictDictOut(MyEstimator.LabelFrame):
    class Config:
        to_format = "dict"
        to_format_kwargs = {"orient": "records"}


@app.get("/")
def health() -> dict[str, str]:
    return {"STATUS": "OK"}


@app.post("/predict/", response_model=DataFrame[PredictDictOut])
async def predict(data: Annotated[DataFrame[MyEstimator.InputFrame], Body()]) -> DataFrame[PredictDictOut]:
    o: DataFrame[PredictDictOut] = model.predict(data.copy())
    return o
