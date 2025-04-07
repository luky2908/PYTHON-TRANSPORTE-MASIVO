Este programa representa un sistema de transporte masivo utilizando un grafo, donde las estaciones están conectadas por rutas con un tiempo de viaje y un costo asociado. Para encontrar la mejor ruta entre dos estaciones, se usa una variación del algoritmo de Dijkstra, el cual emplea una cola de prioridad para procesar primero los caminos más cortos o económicos.

El sistema permite:

Agregar conexiones entre estaciones con su respectivo tiempo y costo.
Buscar la mejor ruta optimizando el tiempo y/o el costo del viaje.
Mostrar la ruta óptima, junto con el tiempo total y el costo del recorrido.
Este enfoque es útil para modelar redes de transporte, como sistemas de metro, autobuses o trenes, y facilitar la planificación de rutas eficientes para los usuarios.

ACTIVIDAD 3: MÉTODOS DE APRENDIZAJE SUPERVISADO

Proyecto: Transporte Masivo con Inteligencia Artificial

Este proyecto demuestra cómo se puede aplicar el aprendizaje supervisado, específicamente usando árboles de decisión, para clasificar rutas dentro de un sistema de transporte masivo.

Objetivo:

El objetivo principal es predecir si una ruta es rápida, intermedia o lenta, basándose en características como el tiempo de viaje, el costo y si la ruta es directa o no.

Dataset:

Para este proyecto se construyó un dataset simulado que contiene información sobre diferentes rutas. Los datos incluidos son:

Tiempo (en minutos)

Costo (en pesos)

Si la ruta es directa (Sí o No)

La clase de la ruta, que es la etiqueta que se desea predecir (Rápida, Intermedia o Lenta)

Modelo:

El modelo utilizado es un árbol de decisión, implementado con la librería scikit-learn de Python.
Este tipo de modelo permite tomar decisiones claras y visuales basadas en los valores de los atributos del dataset.

Requisitos para ejecutar el modelo:

Antes de correr el programa, se deben instalar las siguientes librerías de Python:

pandas

scikit-learn


ACTIVIDAD 4: MÉTODOS DE APRENDIZAJE NO SUPERVISADO

Proyecto: Transporte Masivo con Inteligencia Artificial
En esta actividad se explora cómo utilizar técnicas de aprendizaje no supervisado, específicamente el algoritmo de agrupamiento KMeans, para analizar patrones ocultos en rutas de transporte masivo.

El objetivo principal es descubrir grupos (clusters) de rutas similares en un sistema de transporte, sin necesidad de contar con etiquetas previas. Esto permite identificar patrones útiles como rutas más rápidas, más económicas o con características operativas comunes.

Modelo:
El modelo utilizado es el algoritmo KMeans, parte de la biblioteca scikit-learn de Python.
Este método agrupa rutas con características similares en diferentes clusters (grupos).

En este caso, se configuró el modelo para formar 3 clusters, con la intención de identificar agrupaciones de rutas según su tiempo, costo, paradas y si son directas o no.


