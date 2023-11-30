from fastapi import FastAPI
from pydantic import create_model, BaseModel
from analysis.model import MyModel
from typing import Union, List
import pandas as pd
from decouple import config
from helper.utils import get_categories
import os
from fastapi.encoders import jsonable_encoder

categories_dict = get_categories(pd.read_csv(os.path.join(config("DATA_PATH"), "X.csv"), index_col=0))


Item = create_model("ItemBaseModel", **categories_dict)
    

class ItemList(BaseModel):
    data: List[Item]


app = FastAPI()
model = MyModel(trained=True)



@app.get("/train_model")
async def train_model():
    model.train_model()
    return {"message": "Training done!"}


@app.post("/predict_label")
async def predict_label(item: Item):
    item = pd.DataFrame([jsonable_encoder(item)])
    label = model.predict_label(item)
    label = label.tolist()
    return jsonable_encoder(label)


@app.get("/test")
async def test():
    return {"message": "I'm running!"}


@app.post("/predict_labels")
async def predict_labels(data: ItemList):
    print(jsonable_encoder(data))
    data = pd.DataFrame(jsonable_encoder(data)["data"])
    print(data)
    labels = model.predict_label(data)
    labels = labels.tolist()
    return jsonable_encoder(labels)