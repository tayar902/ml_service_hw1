from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
from typing import List
import pickle
import numpy as np
import pandas as pd
from io import StringIO


app = FastAPI()


class Item(BaseModel):
    name: str
    year: int
    selling_price: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str
    engine: str
    max_power: str
    torque: str
    seats: float


class Items(BaseModel):
    objects: List[Item]


with open('model_scaler.pickle', 'rb') as f:
    pickle_data = pickle.load(f)

model = pickle_data['model']
scaler = pickle_data['scaler']


def prepare_df(df):
    df = df.copy()
    df = df.dropna()
    df['mileage'] = df['mileage'].str.replace(
        ' kmpl', '').str.replace(' km/kg', '')
    df['mileage'] = df['mileage'].replace('', np.nan)
    df['mileage'] = df['mileage'].astype(float)
    df['engine'] = df['engine'].str.replace(' CC', '')
    df['engine'] = df['engine'].replace('', np.nan)
    df['engine'] = df['engine'].astype(float)
    df['max_power'] = df['max_power'].str.replace(' bhp', '')
    df['max_power'] = df['max_power'].replace('', np.nan)
    df['max_power'] = df['max_power'].astype(float)
    df.drop(columns=['torque'], inplace=True)
    df['engine'] = df['engine'].astype(int)
    df['seats'] = df['seats'].astype(int)
    df = df.select_dtypes(include=["float64", "int64"])
    return df


@app.post("/predict_item", description='Метод для предсказания цены одного автомобиля')
def predict_item(items: Items) -> float:
    items_list = [item.dict() for item in items.objects]
    df = pd.DataFrame(items_list)
    try:
        df = prepare_df(df)
    except ValueError:
        raise HTTPException(400, 'Некорректные данные')
    df = df.drop(columns=['selling_price'])
    df_scaled = scaler.transform(df)
    prediction = model.predict(df_scaled)
    return prediction[0]


@app.post("/predict_items", description='Метод для предсказания цен автомобилей из CSV-файла')
async def predict_file(file: UploadFile = File(...)) -> str:
    contents = await file.read()
    df = pd.read_csv(StringIO(contents.decode('utf-8')), index_col=0)
    try:
        df = prepare_df(df)
    except ValueError:
        raise HTTPException(400, 'Некорректные данные')
    prices = df['selling_price']
    df = df.drop(columns=['selling_price'])
    df_scaled = scaler.transform(df)
    predictions = model.predict(df_scaled)
    df['predicted_price'] = predictions.astype(int)
    df['real_price'] = prices
    output_file = 'predictions.csv'
    df.to_csv(output_file, index=False)

    return output_file
