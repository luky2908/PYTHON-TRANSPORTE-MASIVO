import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Cargar el dataset
df = pd.read_csv("dataset_no_supervisado.csv")

# Seleccionar características
X = df[['tiempo', 'costo', 'paradas', 'directa']]

# Escalamiento de datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Modelo KMeans
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_scaled)

# Añadir etiquetas al dataset
df['cluster'] = kmeans.labels_

# Mostrar resultados
print("\nAgrupamientos del sistema de transporte masivo:")
print(df)

# Visualización simple de los clusters
plt.scatter(df['tiempo'], df['costo'], c=df['cluster'], cmap='viridis')
plt.xlabel("Tiempo de viaje (min)")
plt.ylabel("Costo (pesos)")
plt.title("Clusters de rutas de transporte")
plt.colorbar(label='Cluster')
plt.grid(True)
plt.show()
