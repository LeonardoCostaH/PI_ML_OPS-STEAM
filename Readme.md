<p align=center><img src=https://d31uz8lwfmyn8g.cloudfront.net/Assets/logo-henry-white-lg.png><p>

# <h1 align=center> **PROYECTO INDIVIDUAL Nº1** </h1>

# <h1 align=center>**`Machine Learning Operations (MLOps)`**</h1>

<p align="center">
<img src="https://user-images.githubusercontent.com/67664604/217914153-1eb00e25-ac08-4dfa-aaf8-53c09038f082.png"  height=300>
</p>

¡Bienvenidos a mi proyecto de recomendación de juegos en Steam!.  

<hr>  
## Descripción del Proyecto

Este proyecto se enfoca en la creación de un sistema de recomendación de videojuegos para usuarios de la plataforma Steam. La idea principal es ayudar a los usuarios a descubrir nuevos juegos que podrían interesarles, lo que a su vez puede aumentar la retención de usuarios y las ventas en la plataforma.

## Ciclo de Vida del Proyecto

### Rol Desarrollado

En este proyecto, asumí los roles de **Data Scientist** y **Data Engineer**. Comencé desde cero, abordando los desafíos de los datos, realizando análisis exploratorio, creando modelos de recomendación y finalmente implementando una API para servir las recomendaciones a los usuarios.

## Propuesta de Trabajo

El proyecto se divide en varias etapas, que incluyen:

### Transformaciones de Datos

Lectura y procesamiento del dataset con el formato correcto. Se eliminaron columnas innecesarias para optimizar el rendimiento de la API y el entrenamiento del modelo.

### Feature Engineering

Se aplicó análisis de sentimiento con NLP para categorizar reseñas de juegos en tres categorías: malo, neutral y positivo. Esto facilita el trabajo de los modelos de machine learning y el análisis de datos.

### Desarrollo de la API

Se utilizó el framework FastAPI para exponer los datos y las consultas a los usuarios. Las consultas incluyen información sobre desarrolladores, usuarios, géneros y juegos recomendados.

### Deployment

El proyecto se puede implementar en servicios como Render o Railway para que la API esté disponible en línea.

### Análisis Exploratorio de Datos (EDA)

Se realizó un análisis exploratorio de los datos para comprender las relaciones entre las variables del dataset, identificar outliers y descubrir patrones interesantes.

### Modelo de Aprendizaje Automático

Se desarrolló un modelo de recomendación de juegos basado en la similitud del coseno. Los usuarios pueden obtener recomendaciones de juegos similares a uno que les guste.

## Video de Presentación

[https://youtu.be/LC7SsyMR-oc](#) - El video muestra las consultas propuestas en funcionamiento desde la API y una breve explicación del modelo de recomendación.

[https://leoc-pi-ml-ops-steam.onrender.com](#) - Enlace de mí MVP para consumo

¡Gracias por visitar mi proyecto! :rocket:


## Tecnologías Utilizadas

### Lenguaje de Programación

- **Python**: Lenguaje principal utilizado para el desarrollo de este proyecto.
- **Jupyter Notebook**: Utilizado para la exploración de datos y desarrollo de código interactivo.

### Bibliotecas y Frameworks

- **Pandas**: Utilizado para la manipulación y análisis de datos.
- **NumPy**: Fundamental para operaciones matriciales y cálculos numéricos.
- **Scikit-Learn**: Empleado en la creación y entrenamiento de modelos de machine learning.
- **FastAPI**: Framework utilizado para construir la API que sirve recomendaciones a los usuarios.
- **TextBlob**: Utilizado para análisis de sentimiento en las reseñas de los usuarios.

### Procesamiento de Lenguaje Natural (NLP)

- **TextBlob**: Utilizado para tareas de procesamiento de lenguaje natural, como análisis de sentimiento en las reseñas de los usuarios.

### Despliegue

- **Render**: Plataforma de despliegue en la nube utilizada para alojar la API y permitir su acceso en línea.

### Bibliotecas de Machine Learning

- **Scikit-Learn**: Utilizado para implementar el modelo de recomendación basado en la similitud del coseno.

### Almacenamiento de Datos

- **JSON**: Formato utilizado para almacenar datos en ciertas etapas del proyecto.
- **CSV**: Formato empleado para la exportación e importación de datos.
- **Parquet**: Formato de archivo columnar utilizado para el almacenamiento eficiente de datos.

## Uso

Proporciona instrucciones básicas sobre cómo usar tu proyecto. Por ejemplo, cómo ejecutar la API, cómo obtener recomendaciones, etc.

## Licencia

Este proyecto está bajo la Licencia XYZ. Consulta el archivo LICENSE.md para obtener más detalles.

## Contacto

- **Nombre**: Leonardo Augusto Costa Hermes
- **Correo Electrónico**: lcostahermes@gmail.com
- **LinkedIn**: [linkedin.com/in/leonardo-costa-672a3a1b9](https://www.linkedin.com/in/leonardo-costa-672a3a1b9)
