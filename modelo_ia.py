import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# Cargar dataset
data = pd.read_csv("dataset_transporte.csv")

# Codificar variables categóricas
le_ruta = LabelEncoder()
le_clase = LabelEncoder()

data['ruta_directa'] = le_ruta.fit_transform(data['ruta_directa'])  # Sí/No → 1/0
data['clase_ruta'] = le_clase.fit_transform(data['clase_ruta'])     # Rápida, etc.

# Separar X (características) e y (etiqueta)
X = data[['tiempo', 'costo', 'ruta_directa']]
y = data['clase_ruta']

# Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Crear y entrenar el modelo
modelo = DecisionTreeClassifier(criterion='entropy', random_state=0)
modelo.fit(X_train, y_train)

# Hacer predicciones
y_pred = modelo.predict(X_test)

# Evaluar
print("Precisión:", accuracy_score(y_test, y_pred))
print("\nReporte de clasificación:\n")
print(classification_report(
    y_test,
    y_pred,
    labels=[0, 1, 2],
    target_names=le_clase.classes_
))

