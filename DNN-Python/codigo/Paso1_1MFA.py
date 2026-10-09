import numpy as np
import matplotlib.pyplot as plt

# Discretización del dominio espacial y temporal
z_min, z_max = 0, 50  # Límites de la pila de lixiviación en metros (z va de 0 a 50 m)
num_points = 50  # Número de puntos espaciales en la pila (discretización de la altura z)
H = 50  # Altura de la pila de lixiviación en metros
T_max = 200  # Número de pasos temporales para la simulación (aún no usado aquí)
'''
adimensionalizaciones para obtener una ecuación de Richards sin unidades.
ξ = z/H ; ∀z ϵ [0, H] ⇒ ξ ϵ [0, 1]
t = T_max * τ ; ∀t ϵ [0, T_max] ⇒ τ ϵ [0, 1]
Se: es la saturación efectiva del suelo. Es una variable adimensional que se utiliza 
para describir la fracción de poros del suelo que están llenos de agua, 
en relación con la capacidad total de retención de agua del suelo.
Se = (θ - θR) / (θS - θR) ; ∀θ ϵ [θR, θS] ⇒ Se ϵ [0, 1]
θ = θR + (θS - θR) * Se(ξ, τ) ; ∀Se ϵ [0, 1] ⇒ θ ϵ [θR, θS]
θR = 0.01 ; θS = 0.4 en Chile para un suelo arenoso
θ = 0.01 + 0.39 * Se(ξ, τ)

La ecuación de Richards es:
𝜕𝜃/𝜕𝑡 = 𝜕/𝜕𝑧 (D(𝜃) 𝜕𝜃/𝜕𝑧 ) - 𝜕/𝜕𝑧 K(𝜃) 
Desarrollando la ecuación de Richards con los términos de segundo orden y primer orden con la regla de la cadena:
𝜕𝜃/𝜕𝑡 = 𝜕/𝜕𝑧 (D(𝜃) 𝜕𝜃/𝜕𝑧 ) - 𝜕/𝜕𝑧 K(𝜃) = 𝜕𝐷(𝜃)/𝜕𝑧 * 𝜕𝜃/𝜕𝑧 + D(𝜃) * ∂²θ/∂z² - 𝜕K(θ)/∂z (0)

Donde:
θ: Contenido de humedad.
D(θ): Término difusivo relacionado con la conductividad hidráulica K(θ) y el gradiente de humedad 
K(θ): Conductividad hidráulica, dependiente del contenido de humedad.
z: Coordenada axial.
t: Tiempo.

Condiciones de borde:
En z = 0 (superficie de la pila):
-D(θ)∂θ/∂z + K(θ) = R(t)
Donde 
R(t) es la tasa de riego dependiente del tiempo.
z = H (base de la pila):
∂θ/∂z = 0
Condición inicial para todo z∈[0,H]:
θ=θini

Para la discretización de la ecuación de Richards, se utiliza el método de diferencias finitas.
Fórmulas de diferencias finitas para la ecuación de Richards (no adimensionalizada):
∂θ/∂t ≈ (θ[i,j+1] - θ[i,j]) / Δt donde Δt = T_max / num_steps y θ[i,j] : humedad en el punto i en el tiempo j (1)
derivada parcial de primer orden se aproximan con diferencias finitas centradas:
∂D(θ)/∂z ≈ (D(θ[i+1,j]) - D(θ[i-1,j]) / 2Δz donde Δz = H / num_points y θ[i,j] : humedad en el punto i en el tiempo j (2)
derivada parcial de primer orden se aproximan con diferencias finitas centradas:
∂θ/∂z ≈ (θ[i+1,j] - θ[i-1,j]) / 2Δz donde θ[i,j] : humedad en el punto i en el tiempo j (3)
derivada parcial de primer orden se aproximan con diferencias finitas centradas:
∂K(θ)/∂z ≈ (K(θ[i+1,j]) - K(θ[i-1,j]) / 2Δz donde θ[i,j] : humedad en el punto i en el tiempo j (4)
Para la segunda derivada de θ, se emplea una diferencia finita centrada de segundo orden:
∂²θ/∂z² ≈ (θ[i+1,j] - 2*θ[i,j] + θ[i-1,j]) / Δz² donde θ[i,j] : humedad en el punto i en el tiempo j (5)
Sustituyendo estas aproximaciones en la ecuación (0), se obtiene:
(θ[i,j+1] - θ[i,j]) / Δt ≈ (D(θ[i+1,j]) - D(θ[i-1,j]) / 2Δz * (θ[i+1,j] - θ[i-1,j]) / 2Δz + D(θ[i,j]) * (θ[i+1,j] - 2*θ[i,j] + θ[i-1,j]) / Δz²  - (K(θ[i+1,j]) - K(θ[i-1,j]) / 2Δz (6)
Despejando θ[i,j+1] de la ecuación (6):

θ[i,j+1] = θ[i,j] + Δt * [(D(θ[i+1,j]) - D(θ[i-1,j]) / 2Δz * (θ[i+1,j] - θ[i-1,j]) / 2Δz + D(θ[i,j]) * (θ[i+1,j] - 2*θ[i,j] + θ[i-1,j]) / Δz²  - (K(θ[i+1,j]) - K(θ[i-1,j]) / 2Δz] (7)
Esta expresión permite calcular el valor de θ en la nueva iteración temporal j+1 en función de los valores de la iteración actual j.

Condiciones de bordes:
En z = 0 (superficie de la pila):
-D(θ)∂θ/∂z + K(θ) = R(t) y t > 0
discretizando la condición de borde en z = 0:
-D(θ[1,j]) * (θ[2,j] - θ[0,j]) / 2Δz + K(θ[1,j]) = R[j] y despejando θ[0,j]:
θ[0,j] = θ[2,j] + 2Δz * (R[j] - K(θ[1,j])) / D(θ[1,j]) (8) para todo j∈[1, T_max]
En z = H (base de la pila):
∂θ/∂z = 0
discretizando la condición de borde en z = H:
(θ[N,j] - θ[N-1,j]) / Δz = 0 y despejando θ[N,j]:
θ[N,j] = θ[N-1,j] (9) donde N = num_points - 1 , θ[N,j] : humedad en el punto N en el tiempo j y θ[N-1,j] : humedad en el punto N-1 en el tiempo j
si la altura de la pila es de 50 metros, entonces N = 49.
condición inicial para todo z∈[0,H]:
θ=θini y j = 0 (10) donde θini es la humedad inicial en la pila de lixiviación y 
θini = 0.01 + 0.39 * Se(ξ, 0), donde Se(ξ, 0) es la saturación efectiva en el tiempo inicial j = 0.
'''
# Crear el dominio espacial (discretización de z)
z_discreto = np.linspace(z_min, z_max, num_points)  # 50 puntos equidistantes entre z_min y z_max
# Imprimir z_discreto
print("Coordenadas espaciales discretizadas (z_discreto):")
print(z_discreto)
# Inicializar la condición inicial x(t=0, z)
# según condición inicial para todo z∈[0,H]: θ=θini donde θini = 0.01 + 0.39 * Se(ξ, 0)
def inicializar_x(z):
    #x = 0.4 * np.exp(-2 * z) + 0.2  # Distribución exponencial
    x = 0.01 + 0.39 * np.exp(-0.1 * z)
    #return (x - np.min(x)) / (np.max(x) - np.min(x))  # Normalizar entre 0 y 1
    # Distribución exponencial basado en Van Genuchten (1980) y Tracy (2006)
    return x
# imprimir np.min(x) y np.max(x)
print("Minimo y maximo de x")
x = 0.01 + 0.39 * np.exp(-0.1 * z_discreto)
print(np.min(x))
print(np.max(x))

x_0 = inicializar_x(z_discreto)  # Generar los valores iniciales de x(t=0, z)
# Impresión de los parámetros configurados

print("Parámetros de simulación configurados:")
print(f"Número máximo de pasos temporales (T_max): {T_max}")
print(f"Dominio espacial: [{z_min}, {z_max}] con {num_points} puntos discretos")
print(f"Paso temporal (dt): No definido para esta parte del código")
# Imprimir los datos iniciales para t=0
print("Datos iniciales (t=0):")
for i, valor in enumerate(x_0):
    print(f"z[{i}]: {z_discreto[i]:.2f} m, x_0[{i}]: {valor:.4f}")
# Graficar los datos iniciales
plt.figure(figsize=(8, 4))
plt.plot(z_discreto, x_0, marker='o', label='x_0 (t=0)')
plt.title('Distribución inicial x(t=0, z) en la pila de lixiviación para todo z∈[0,H]')
plt.xlabel('z (Altura en metros)')
plt.ylabel('x_0 (Contenido de humedad inicial)')
plt.grid()
plt.legend()
plt.show()
