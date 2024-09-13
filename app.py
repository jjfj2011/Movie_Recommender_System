import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.neighbors import NearestNeighbors
from recommendations import preprocesamiento, recomendacion_populares_avanzada, recomendacion_jaccard, recomendacion_tf_idf, recomendacion_knn

# Se cargan los datos
# @st.cache
def cargar_datos():
    df_movies = pd.read_csv('movies.csv')
    df_ratings = pd.read_csv('ratings.csv')
    return df_movies, df_ratings

df_movies, df_ratings = cargar_datos()

# Se procesan los datos
preprocesamiento(df_movies, df_ratings)

#####################################################################################################################################
# Filtro Colaborativo
# Creamos la matriz de usuario-película
ratings_matrix = df_ratings.pivot_table(index='userId', columns='movieId', values='rating')
ratings_matrix_normalized = ratings_matrix.sub(ratings_matrix.mean(axis=1), axis=0).fillna(0)

# Se crea y entrenamos el modelo KNN
knn_model = NearestNeighbors(metric='cosine', algorithm='brute')
knn_model.fit(ratings_matrix_normalized)
#####################################################################################################################################

# Creamos la interfaz de usuario en Streamlit
st.title("Sistema de Recomendación de Películas")

st.sidebar.image('logo.png')

# Mostramos las opciones de recomendación
opcion = st.sidebar.selectbox(
    "Elige el tipo de recomendación",
    ["Recomendación Popular", "Recomendación Basada en Contenido - Jaccard", "Recomendación Basada en Contenido - TF-IDF", "Recomendación Basada en Filtro Colaborativo"]
)

# Recomendación Popular
if opcion == "Recomendación Popular":
    st.subheader("Recomendación de Películas Populares")
    df_final = df_ratings.merge(df_movies[['movieId', 'title', 'genres']], on='movieId', how='left')
    top_10_movies = recomendacion_populares_avanzada(df_movies, df_ratings, df_final)
    #st.write(top_10_movies)
    st.dataframe(top_10_movies, hide_index=True, use_container_width=True)

# Recomendación Basada en Contenido - Jaccard
elif opcion == "Recomendación Basada en Contenido - Jaccard":
    st.subheader("Recomendación de Películas Basada en Contenido - Similitud de Jaccard")
    
    # Selección del título de la película
    movie_title = st.selectbox("Selecciona el título de la película", df_movies['title'].unique())

    if st.button("Obtener Recomendaciones"):
        recomendaciones = recomendacion_jaccard(movie_title, df_movies)
        #st.write(recomendaciones)
        st.dataframe(recomendaciones, hide_index=True, use_container_width=True)

# Recomendación Basada en Contenido - TF-IDF
elif opcion == "Recomendación Basada en Contenido - TF-IDF":
    st.subheader("Recomendación de Películas Basada en Contenido - Similitud TF-IDF")
    
    # Selección del título de la película
    movie_title = st.selectbox("Selecciona el título de la película", df_movies['title'].unique())

    if st.button("Obtener Recomendaciones"):
        # Creamos la matriz TF-IDF
        tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf_vectorizer.fit_transform(df_movies['content'])
        cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
        
        # Llama a la función de recomendación con el título seleccionado
        recomendaciones = recomendacion_tf_idf(movie_title, df_movies, cosine_sim)
        #st.write(recomendaciones)
        st.dataframe(recomendaciones, hide_index=True, use_container_width=True)

# Recomendación Basada en Filtro Colaborativo
elif opcion == "Recomendación Basada en Filtro Colaborativo":
    st.subheader("Recomendación Basada en Filtro Colaborativo")

    # Selecciona si es un usuario nuevo o existente
    usuario_tipo = st.radio("¿Eres un usuario nuevo o existente?", ("Nuevo Usuario", "Usuario Existente"))

    # Para el nuevo usuario
    if usuario_tipo == "Nuevo Usuario":
        # Selecciona el título de la película
        movie_title = st.selectbox("Selecciona el título de la película", df_movies['title'].unique())

        # Selecciona la calificación que oscila entre 1 y 5
        user_rating = st.slider("Selecciona tu calificación para esta película", 1, 5)

        if st.button("Obtener Recomendaciones"):
            # Obtenemos el movieId correspondiente al título seleccionado
            movie_id = df_movies[df_movies['title'] == movie_title]['movieId'].values[0]

            # Creamos una nueva serie con el movieId y la calificación del nuevo usuario
            nuevo_usuario_ratings = pd.Series({movie_id: user_rating})

            # Genera recomendaciones utilizando la función recomendacion_knn
            recomendaciones = recomendacion_knn(nuevo_usuario_ratings, ratings_matrix_normalized, ratings_matrix, df_movies, knn_model)
            
            # Mostramos las recomendaciones en la interfaz de Streamlit
            st.dataframe(recomendaciones, hide_index=True, use_container_width=True)

    # Para el usuario existente
    else:
        usuario_id = st.number_input("Ingresa tu ID de usuario", min_value=1, max_value=df_ratings['userId'].max())

        if st.button("Obtener Recomendaciones"):
            # Genera recomendaciones utilizando la función recomendacion_knn para el usuario existente
            recomendaciones = recomendacion_knn(usuario_id, ratings_matrix_normalized, ratings_matrix, df_movies, knn_model)

            # Muestra las recomendaciones en la interfaz de Streamlit
            st.dataframe(recomendaciones, hide_index=True, use_container_width=True)