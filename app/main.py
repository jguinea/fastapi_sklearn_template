from fastapi import FastAPI
from pydantic import create_model, BaseModel
from analysis.model import MyModel
from typing import Union, List
import pandas as pd
from decouple import config
from helper.utils import get_categories
import os
from fastapi.encoders import jsonable_encoder


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
async def predict_label(item: InputItem) -> OutputItem:
    item = pd.DataFrame([jsonable_encoder(item)])
    label = model.predict_label(item)
    label = label.tolist()
    return jsonable_encoder(label)


@app.get("/test")
async def test():
    return {"message": "I'm running!"}


@app.post("/predict_labels")
async def predict_labels(data: list[InputItem]) -> OutputItem:
    print(jsonable_encoder(data))
    data = pd.DataFrame(jsonable_encoder(data)["data"])
    print(data)
    labels = model.predict_label(data)
    labels = labels.tolist()
    return jsonable_encoder(labels)