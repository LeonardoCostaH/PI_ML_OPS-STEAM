from fastapi import FastAPI
import pandas as pd
import numpy as np
from typing import Optional
import sklearn
from sklearn.metrics.pairwise import cosine_similarity
from ML import recomendacion_juego

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

@app.get("/developer1/{desarrollador}")
async def developer(desarrollador: str):
    # Filtrar los juegos del desarrollador especificado
    juegos_del_desarrollador = df_games[df_games['developer'] == desarrollador]

    if juegos_del_desarrollador.empty:
        return {"error": "No se encontraron juegos para el desarrollador especificado."}

    # Combinar los DataFrames para obtener los juegos de ese desarrollador
    combined_df = pd.merge(df_items, juegos_del_desarrollador, on="item_id", how="inner")

    # Calcular el porcentaje de juegos gratuitos y reemplazar los valores nulos con 0.0
    contenido_free_por_año = (
        combined_df[combined_df['price'] == 0.0]
        .groupby('year')['item_id']
        .count()
        .div(
            combined_df.groupby('year')['item_id']
            .count()
            .fillna(0.0)  # Reemplazar valores nulos con 0.0
        )
        .mul(100)
        .reset_index()
    )
    contenido_free_por_año.columns = ['Año', 'Contenido Free']

    # Asegúrate de que cualquier valor NaN en 'Contenido Free' se reemplace con 0.0
    contenido_free_por_año['Contenido Free'].fillna(0.0, inplace=True)

    # Aplicar el formato con "%" a la columna 'Contenido Free'
    contenido_free_por_año['Contenido Free'] = contenido_free_por_año['Contenido Free'].apply(lambda x: '{:.2f}%'.format(x))

    # Calcular la cantidad de elementos por año
    cantidad_por_año = combined_df.groupby('year')['item_id'].count().reset_index()
    cantidad_por_año.columns = ['Año', 'Cantidad de Items']

    # Combinar los DataFrames de cantidad y contenido gratuito
    resultado = pd.merge(cantidad_por_año, contenido_free_por_año, on="Año", how="left")

    # Convertir el DataFrame de resultado a un diccionario de diccionarios
    resultado_dict = resultado.set_index('Año').to_dict(orient='index')

    return resultado_dict







@app.get("/userdata/{user_id}")
def userdata(user_id:str):
    # Convierte user_id a tipo str
    user_id = str(user_id)    
    # Filtra las compras del usuario en df_items
    compras_usuario = df_items[df_items['user_id'] == user_id]
    # Combina la información de las compras con los datos de los juegos en df_games
    compras_usuario = pd.merge(compras_usuario, df_games, on='item_id', how='inner')
    # Calcula el gasto total del usuario
    gasto_total = compras_usuario['price'].sum()
    # Filtra las revisiones del usuario en df_reviews
    revisiones_usuario = df_reviews[(df_reviews['user_id'] == user_id) & (df_reviews['item_id'].isin(compras_usuario['item_id']))]
    # Calcula el porcentaje de recomendación positiva
    porcentaje_recomendacion = (revisiones_usuario['recommend'].sum() / len(revisiones_usuario)) * 100
    # Calcula la cantidad de ítems comprados
    cantidad_items = len(compras_usuario)
    # Devuelve las estadísticas
    return {
        'Gasto Total': round(gasto_total,2),
        'Porcentaje de Recomendación Promedio': porcentaje_recomendacion,
        'Cantidad de Ítems': cantidad_items
    }


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



@app.get("/best_developer_year/{year}")
async def best_developer_year(año: int):
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





@app.get("/developer2/{desarrolladora}")
def developer(desarrolladora: str):
    # Filtrar las reseñas por el desarrollador dado
    reseñas_desarrolladora = df_reviews[df_reviews['user_id'].isin(df_items[df_items['item_id'].isin(df_games[df_games['developer'] == desarrolladora]['item_id'])]['user_id'])]

    # Contar las reseñas con sentimiento positivo y negativo
    sentimiento_positivo = reseñas_desarrolladora[reseñas_desarrolladora['sentiment_analysis'] == 2].shape[0]
    sentimiento_negativo = reseñas_desarrolladora[reseñas_desarrolladora['sentiment_analysis'] == 0].shape[0]

    # Crear el diccionario de resultados
    resultado = {desarrolladora: {'Positive': sentimiento_positivo, 'Negative': sentimiento_negativo}}

    return resultado


@app.get("/recomendacion_juego/{game_id}")
def get_recommendation(game_id: int):
    recommendations = recomendacion_juego(game_id)
    return recommendations