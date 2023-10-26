from fastapi import FastAPI
import json
import ast
import re
from textblob import TextBlob
import pandas as pd
import numpy as np
from typing import Optional
import sklearn
from sklearn.metrics.pairwise import cosine_similarity

ruta_parquet_games = "datasets/data_steam_games.parquet"
ruta_parquet_reviews = "datasets/data_reviews.parquet"
ruta_parquet_items = "datasets/data_items.parquet"

df_games = pd.read_parquet(ruta_parquet_games)
df_items = pd.read_parquet(ruta_parquet_items)
df_reviews = pd.read_parquet(ruta_parquet_reviews)

app = FastAPI()

# http://127.0.0.1:8000

@app.get("/")
def index():
    return "Bienvenidos a mí proyecto"




@app.get("/UserForGenre/{genero}")
def UserForGenre(genero: str):
    # Filtrar df_games para obtener solo las filas que contienen el género especificado
    filtered_games = df_games[df_games['genres'].str.contains(genero, case=False, na=False)]

    # Filtrar df_items para reducir el conjunto de datos a las columnas necesarias
    df_items_filtered = df_items[['user_id', 'item_id', 'playtime_forever']]

    # Combinar los DataFrames filtrados en uno solo usando "item_id" como clave
    combined_df = pd.merge(df_items_filtered, filtered_games, on="item_id", how="inner")

    # Agrupar por usuario y año, sumar las horas jugadas y encontrar el usuario con más horas jugadas
    result_df = combined_df.groupby(['user_id', 'year'])['playtime_forever'].sum().reset_index()
    max_user = result_df.loc[result_df['playtime_forever'].idxmax()]

    # Convertir las horas jugadas de minutos a horas en el DataFrame resultante
    result_df['playtime_forever'] = result_df['playtime_forever'] / 60
    result_df['playtime_forever'] = result_df['playtime_forever'].round(0)

    # Crear una lista de acumulación de horas jugadas por año
    accumulation = result_df.groupby('year')['playtime_forever'].sum().reset_index()
    accumulation = accumulation.rename(columns={'year': 'Año', 'playtime_forever': 'Horas'})
    accumulation_list = accumulation.to_dict(orient='records')

    return {"Usuario con más horas jugadas para el género " + genero: max_user['user_id'], "Horas jugadas": accumulation_list}


@app.get("/best_developer_year/{año}")
def best_developer_year(año: int):
    # Filtra los juegos del año especificado en df_games
    juegos_del_año = df_games[df_games['year'] == año]

    # Combinación de DataFrames para obtener los juegos recomendados en ese año
    combined_df = pd.merge(juegos_del_año, df_reviews, on="item_id", how="inner")

    # Filtra los juegos recomendados con comentarios positivos
    juegos_recomendados = combined_df[(combined_df['recommend'] == True) & (combined_df['sentiment_analysis'] == 2)]

    # Agrupa por desarrollador y cuenta las recomendaciones
    desarrolladores_recomendados = juegos_recomendados['developer'].value_counts().reset_index()
    desarrolladores_recomendados.columns = ['developer', 'recommend_count']

    # Ordena en orden descendente y toma los 3 principales
    top_desarrolladores = desarrolladores_recomendados.nlargest(3, 'recommend_count')

    # Formatea el resultado en un formato de lista de diccionarios
    resultado = [{"Puesto {}: {}".format(i + 1, row['developer']): row['recommend_count']} for i, row in top_desarrolladores.iterrows()]

    return resultado



@app.get("/developer/{desarrolladora}")
def developer(desarrolladora: str):
    # Filtrar las reseñas por el desarrollador dado
    reseñas_desarrolladora = df_reviews[df_reviews['user_id'].isin(df_items[df_items['item_id'].isin(df_games[df_games['developer'] == desarrolladora]['item_id'])]['user_id'])]

    # Contar las reseñas con sentimiento positivo y negativo
    sentimiento_positivo = reseñas_desarrolladora[reseñas_desarrolladora['sentiment_analysis'] == 2].shape[0]
    sentimiento_negativo = reseñas_desarrolladora[reseñas_desarrolladora['sentiment_analysis'] == 0].shape[0]

    # Crear el diccionario de resultados
    resultado = {desarrolladora: {'Positive': sentimiento_positivo, 'Negative': sentimiento_negativo}}

    return resultado

