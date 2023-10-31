import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer
import ast

ruta_parquet_games = "datasets/data_steam_games.parquet"
ruta_parquet_reviews = "datasets/data_reviews.parquet"
ruta_parquet_items = "datasets/data_items.parquet"

df_games = pd.read_parquet(ruta_parquet_games)


# Crear una copia del DataFrame para evitar modificar el original
df_steam_copy = df_games.copy()

# Desanidar las columnas de género
df_steam_copy['genres'] = df_steam_copy['genres'].apply(eval)  # Convierte las listas de géneros en listas de Python
unique_genres = list(set(genre for sublist in df_steam_copy['genres'] for genre in sublist))

# Crear columnas binarias para cada género
mlb = MultiLabelBinarizer()
genre_features = pd.DataFrame(mlb.fit_transform(df_steam_copy['genres']), columns=mlb.classes_)

def recomendacion_juego(id_producto, num_recomendaciones=5):
    # Verificar si el ID del juego está en el índice de genre_features
    if id_producto not in genre_features.index:
        return {"error": "El juego no se encuentra en la base de datos."}

    # Crear una copia del DataFrame de género y eliminar el juego seleccionado
    genre_features_copy = genre_features.copy()
    juego_seleccionado = genre_features_copy.loc[id_producto]
    genre_features_copy.drop(id_producto, inplace=True)

    # Resto del código para calcular la similitud del coseno y obtener las recomendaciones
    juego_seleccionado = np.array(juego_seleccionado.values).reshape(1, -1)
    similaridades = cosine_similarity(juego_seleccionado, genre_features_copy)

    # Encuentra los juegos más similares sin incluir el juego seleccionado
    juegos_similares_indices = similaridades.argsort()[0][-num_recomendaciones:][::-1]
    juegos_recomendados = df_steam_copy.iloc[juegos_similares_indices, :]

    lista = [{"item_id": row['item_id'], "title": row['title']} for index, row in juegos_recomendados.iterrows()]

    return lista