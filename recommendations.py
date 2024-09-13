import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.neighbors import NearestNeighbors

def preprocesamiento(df_movies, df_ratings):
    # Para ambos dataframe elimina los registros con alguna columna nula
    df_movies.dropna(inplace=True)
    df_ratings.dropna(inplace=True)

    # Elimina todos los registros duplicados en df_movies, considera la columna 'movieId' como llave
    df_movies.drop_duplicates(subset=['movieId'], inplace=True)

    # Elimina todos los registros duplicados en df_ratings, considera las columnas 'movieId','userId' como llaves
    df_ratings.drop_duplicates(subset=['movieId','userId'], inplace=True)

    # En df_movies puedes crear la columna 'content' a partir de la columna 'genres', apenas reemplazando '|' por ' ', usaremos esta nueva columna más adelante
    df_movies['content'] = df_movies['genres'].str.replace('|', ' ', regex=False)

    # En 'df_movies' puedes crear la columna 'genre_set' a partir de la columna 'genres', esta nueva columna es de tipo set y contiene todos los géneros separados por coma ','
    df_movies['genre_set'] = df_movies['genres'].apply(lambda x: set(x.split('|')))

    # Asegúrate de que todas las columnas, de cada uno de los 2 dataframe, tengan el tipo de datos correcto, números con into float, textos con object y fechas con datetime
    df_ratings['timestamp'] = pd.to_datetime(df_ratings['timestamp'], unit='s')

# Recomendación No Personalizada (Bayesian average)
def recomendacion_populares_avanzada(df_movies, df_ratings, df_final):
    # Agrupa las películas por título y calcula el rating promedio y el número de votos
    average_values = df_final.groupby('title').agg(rating_mean=('rating', 'mean'), vote_count=('rating', 'count'))

    # Define el promedio global (C) y el número mínimo de votos requeridos (m)
    C = average_values['rating_mean'].mean()  # Rating promedio global
    m = average_values['vote_count'].quantile(0.70)  # Número mínimo de votos, aquí usamos el percentil 70

    # Calcula el weighted_score utilizando el promedio ponderado bayesiano
    def weighted_score(x, m=m, C=C):
        v = x['vote_count']  # Número de votos
        R = x['rating_mean']  # Rating promedio
        return (v / (v + m) * R) + (m / (m + v) * C)

    # Se crea una nueva columna 'weighted_score' basada en la fórmula anterior
    average_values['weighted_score'] = average_values.apply(weighted_score, axis=1)

    # Ordenamos por weighted_score y obtenemos las 10 mejores películas
    top_10_movies = average_values.sort_values(by='weighted_score', ascending=False).head(10).reset_index()

    # Merge entre 'top_10_movies' y 'df_movies' para agregar el género
    top_10_movies = top_10_movies.merge(df_movies[['title', 'genres']], on='title', how='left')

    # Reemplazamos '|' por ',' en la columna 'genres'
    top_10_movies['genres'] = top_10_movies['genres'].str.replace('|', ', ', regex=False)

    # Renombramos las columnas
    top_10_movies = top_10_movies.rename(columns={
        'title': 'Título',
        'genres': 'Géneros',
        'rating_mean': 'Rating Promedio',
        'vote_count': 'Número de Votos',
        'weighted_score': 'Puntuación Ponderada'
    })

    return top_10_movies[['Título', 'Géneros']]
#####################################################################################################################################

# RECOMENDACIÓN JACCARD
# Define una función que calcule la similitud de Jaccard entre dos conjuntos(set)
def similitud_jaccard(set1, set2):
    interseccion = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return interseccion / union

# Función de recomendación basada en Jaccard
def recomendacion_jaccard(title: str, df_movies: pd.DataFrame, n_recommendations: int = 10) -> pd.DataFrame:
    # Se obtiene el conjunto de características de entrada para un ítem específico
    input_features = df_movies[df_movies['title'] == title]['genre_set'].values[0]

    # Calcula la similitud entre este ítem y todos los demás ítems usando la función de similitud de Jaccard
    df_movies['similaridad'] = df_movies['genre_set'].apply(lambda x: similitud_jaccard(input_features, x))

    # Ordena los ítems por su similitud, excluyendo el ítem de entrada
    recommendations = df_movies[df_movies['title'] != title].sort_values(by='similaridad', ascending=False)

    # Reemplazamos '|' por ',' en la columna 'genres'
    recommendations['genres'] = recommendations['genres'].str.replace('|', ', ', regex=False)

    # Renombramos las columnas
    recommendations = recommendations.rename(columns={
        'title': 'Título', 
        'genres': 'Géneros'
    })

    return recommendations[['Título', 'Géneros']].head(n_recommendations)
#####################################################################################################################################

# RECOMENDACIÓN COSENO
def recomendacion_tf_idf(title: str, df_movies: pd.DataFrame, cosine_sim: pd.DataFrame, n_recommendations: int = 10) -> pd.DataFrame:
    # Se obtiene el índice de la película dada, usando el título en el DataFrame df_movies
    idx = df_movies[df_movies['title'] == title].index[0]

    # Se obtiene las puntuaciones de similitud de coseno para todas las películas, con respecto a la película dada usando la matriz cosine_sim
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Ordenamos las películas por las puntuaciones de similitud en orden descendente
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Filtramos la película de entrada para asegurar de que no se recomiende a sí misma
    sim_scores = sim_scores[1:n_recommendations + 1]

    # Obtenemos los índices de las películas recomendadas en función de las puntuaciones de similitud
    movie_indices = [i[0] for i in sim_scores]

    # Obtenemos las películas recomendadas a partir de los índices
    df_recomendaciones = df_movies.iloc[movie_indices].copy()
    df_recomendaciones['distance'] = [i[1] for i in sim_scores]

    # Reemplazamos '|' por ',' en la columna 'genres'
    df_recomendaciones['genres'] = df_recomendaciones['genres'].str.replace('|',', ', regex=False)

    # Renombramos las columnas
    df_recomendaciones = df_recomendaciones.rename(columns={
        'title':'Título', 
        'genres':'Géneros'
    })

    return df_recomendaciones[['Título', 'Géneros']]
#####################################################################################################################################

# Recomendación Collaborative Filtering
def recomendacion_knn(usuario_id_o_ratings, ratings_matrix_normalized, ratings_matrix, df_movies, knn_model, n_recommendations=10):
    # Identificamos si el usuario es nuevo o si ya existe
    if isinstance(usuario_id_o_ratings, pd.Series):
        # Convertimos al usuario en un df
        usuario_id_o_ratings = pd.DataFrame(usuario_id_o_ratings).transpose()
        # Usamos reindex para crear las columnas de las películas para el usuario y asignar a su lugar correcto la calificación
        usuario_id_o_ratings_complete = usuario_id_o_ratings.reindex(columns=ratings_matrix.columns)
        # Calculamos el promedio del usuario
        avg_user = usuario_id_o_ratings_complete.mean(axis=1)
        # Normalizamos las calificaciones del nuevo usuario
        usuario_id_o_ratings_normalized = usuario_id_o_ratings_complete.sub(avg_user, axis=0).fillna(0)
        # Obtenemos las distancias y los índices de las películas después de pasar por el modelo KNN
        distances, indices = knn_model.kneighbors(usuario_id_o_ratings_normalized.values, n_neighbors=n_recommendations + 1)

    else:
        # Si isinstance() regresa "False", quiere decir que es un ID de usuario existente
        user_idx = ratings_matrix.index.get_loc(usuario_id_o_ratings)
        # Encontrar las distancias y los índices de los vecinos más cercanos
        distances, indices = knn_model.kneighbors(ratings_matrix_normalized.iloc[user_idx, :].values.reshape(1, -1), n_neighbors=n_recommendations + 1)

    # Ignoramos el propio usuario en el primer resultado
    distances = distances.flatten()[1:]
    indices = indices.flatten()[1:]

    # Obtenemos a los usuarios similares
    similar_users = ratings_matrix_normalized.iloc[indices]

    # Calculamos las calificaciones promedio ponderadas para cada película
    mean_ratings = similar_users.T.dot(distances) / np.sum(distances)

    # Convertimos la información en un data frame
    mean_ratings_df = pd.DataFrame(mean_ratings, index=ratings_matrix.columns, columns=['mean_rating'])
    mean_ratings_df = mean_ratings_df.dropna()  # Eliminar las películas no calificadas

    # Se filtra las películas que el usuario ya ha visto
    if isinstance(usuario_id_o_ratings, pd.DataFrame):
        seen_movies = usuario_id_o_ratings.dropna(axis=1).columns
    else:
        seen_movies = ratings_matrix.loc[usuario_id_o_ratings].dropna().index

    recommendations = mean_ratings_df[~mean_ratings_df.index.isin(seen_movies)]

    # Ordenamos las calificaciones promedio en orden descendente
    recommendations = recommendations.sort_values('mean_rating', ascending=False).head(n_recommendations)

    # Merge entre 'recommendations' y 'df_movies' para obtener el género de la película
    recommendations = recommendations.merge(df_movies[['movieId', 'title', 'genres']], left_index=True, right_on='movieId')

    # Reemplazamos '|' por ',' en la columna 'genres'
    recommendations['genres'] = recommendations['genres'].str.replace('|',', ', regex=False)

    # Renombramos las columnas
    recommendations = recommendations.rename(columns={
        'title':'Título', 
        'genres':'Géneros'
    })

    return recommendations[['Título', 'Géneros']]