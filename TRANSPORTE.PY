import heapq

class TransporteMasivo:
    def __init__(self):
        self.grafo = {}  # Aquí guardamos las estaciones y sus conexiones

    def agregar_conexion(self, origen, destino, tiempo, costo):
        """ Agrega una conexión entre dos estaciones con tiempo y costo """
        if origen not in self.grafo:
            self.grafo[origen] = []
        if destino not in self.grafo:
            self.grafo[destino] = []
        
        self.grafo[origen].append((destino, tiempo, costo))
        self.grafo[destino].append((origen, tiempo, costo))  # Si es bidireccional

    def mejor_ruta(self, inicio, destino):
        """ Encuentra la mejor ruta entre dos estaciones """
        # Usamos una cola de prioridad para siempre tomar el camino más corto
        cola_prioridad = [(0, 0, inicio)]  # (tiempo, costo, estación)
        distancias = {nodo: (float('inf'), float('inf')) for nodo in self.grafo}
        distancias[inicio] = (0, 0)
        padres = {inicio: None}

        while cola_prioridad:
            tiempo_actual, costo_actual, nodo_actual = heapq.heappop(cola_prioridad)

            if nodo_actual == destino:  # Si llegamos al destino, terminamos
                ruta = []
                while nodo_actual:
                    ruta.append(nodo_actual)
                    nodo_actual = padres[nodo_actual]
                return list(reversed(ruta)), tiempo_actual, costo_actual

            for vecino, tiempo, costo in self.grafo[nodo_actual]:
                nueva_distancia = tiempo_actual + tiempo
                nuevo_costo = costo_actual + costo

                # Solo actualizamos si encontramos un camino más corto o más barato
                if nueva_distancia < distancias[vecino][0] or nuevo_costo < distancias[vecino][1]:
                    distancias[vecino] = (nueva_distancia, nuevo_costo)
                    padres[vecino] = nodo_actual
                    heapq.heappush(cola_prioridad, (nueva_distancia, nuevo_costo, vecino))

        return None, float('inf'), float('inf')  # Si no hay ruta

# === Definir el sistema de transporte ===
sistema = TransporteMasivo()

# Agregamos estaciones y sus conexiones (origen, destino, tiempo en min, costo en pesos)
sistema.agregar_conexion("A", "B", 5, 2000)
sistema.agregar_conexion("B", "C", 10, 1500)
sistema.agregar_conexion("A", "C", 15, 2500)
sistema.agregar_conexion("C", "D", 5, 1000)
sistema.agregar_conexion("B", "D", 8, 1800)
sistema.agregar_conexion("D", "E", 12, 2200)
sistema.agregar_conexion("C", "E", 10, 1900)
sistema.agregar_conexion("E", "F", 7, 1200)
sistema.agregar_conexion("B", "F", 14, 2700)

# === Probar el sistema ===
origen = "A"
destino = "F"
ruta, tiempo, costo = sistema.mejor_ruta(origen, destino)

if ruta:
    print(f"Mejor ruta de {origen} a {destino}: {ruta}")
    print(f"Tiempo estimado: {tiempo} minutos")
    print(f"Costo total: ${costo}")
else:
    print(f"No hay ruta disponible entre {origen} y {destino}")
