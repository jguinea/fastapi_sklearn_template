from fastapi import FastAPI
from pydantic import create_model, BaseModel
from analysis.model import MyModel
from typing import Union, List
import pandas as pd
from decouple import config
from helper.utils import get_categories
import os
from fastapi.encoders import jsonable_encoder
import logging

logging.basicConfig(level=logging.WARN)

categories_input = get_categories(pd.read_csv(os.path.join(config("DATA_PATH"), "X.csv"), index_col=0))
categories_output = get_categories(pd.read_csv(os.path.join(config("DATA_PATH"), "y.csv"), index_col=0))


InputItem = create_model("ItemBaseModel", **categories_input)
OutputItem = create_model("ItemBaseModel", **categories_output)
    



app = FastAPI()
model = MyModel(trained=True)


@app.on_event("startup")
def load_model():
    model.load_model()


@app.post("/predict_label")
async def predict(item: InputItem) -> OutputItem:
    logging.debug(f"input={item}")
    item = pd.DataFrame([jsonable_encoder(item)])
    logging.debug(f"formated input={item}")
    label = model.predict(item)
    logging.debug(f"label={label}")
    formated_label = dict(enumerate(label))
    logging.debug(f"formated label={formated_label}")
    return jsonable_encoder(formated_label)


@app.get("/test")
async def test():
    return {"message": "I'm running!"}


@app.post("/predict_labels")
async def predict_labels(data: list[InputItem]) -> list[OutputItem]:
    data = pd.DataFrame(jsonable_encoder(data))
    labels = model.predict(data)
    logging.debug(f'model output={labels}')
    formated_labels = [{ind: label} for ind, label in enumerate(labels)]
    logging.debug(f"formated output={formated_labels}")
    return jsonable_encoder(formated_labels)