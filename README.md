# Sistema de recomendación de películas
## Descripción
Este repositorio contiene un sistema de recomendación de películas, diseñado para mejorar la experiencia del usuario en una plataforma de streaming. Se utiliza varias técnicas de recomendación, incluyendo métodos no personalizados, basados en contenido y colaborativos.
## Problema de negocio
En el contexto de plataformas de streaming, la capacidad de utilizar datos para optimizar la experiencia del usuario es fundamental. El objetivo es aprovechar los datos disponibles para desarrollar un sistema de recomendación de películas personalizado, que proporcione sugerencias precisas y relevantes. Nos encargaremos de analizar estos datos utilizando Python, para construir un recomendador de películas que transformará la manera en que los usuarios descubren nuevo contenido, mejorando significativamente su experiencia y satisfacción
## Sistemas de Recomendación Utilizados
En este proyecto, se han implementado varias estrategias de recomendación para proporcionar sugerencias de películas:

1. **Recomendación No Personalizada**: Se basa en la popularidad de las películas. Se identifican las películas más votadas y con mejores calificaciones promedio para recomendar las más populares a todos los usuarios.

2. **Recomendación Basada en Contenido**:
   - **Similitud de Jaccard**: Utiliza la similitud de Jaccard para medir la coincidencia entre los géneros de las películas. Se recomienda películas que comparten un alto grado de similitud en los géneros con una película dada.
   - **Similitud de Coseno con TF-IDF**: Calcula la similitud entre películas, basándose en la representación de texto de sus descripciones utilizando TF-IDF y similitud de coseno. Esto permite recomendar películas similares en función de su contenido textual.

3. **Recomendación Basada en Filtro Colaborativo**:
   - **KNN (K-Nearest Neighbors)**: Utiliza un modelo de KNN basado en la similitud de coseno para encontrar usuarios similares y recomendar películas en función de las calificaciones de estos usuarios similares. Esta técnica considera las preferencias de otros usuarios con perfiles similares para hacer recomendaciones personalizadas.

Cada uno de estos métodos se han implementado y evaluado para proporcionar recomendaciones precisas y relevantes, mejorando la experiencia general de los usuarios en la plataforma de streaming.
