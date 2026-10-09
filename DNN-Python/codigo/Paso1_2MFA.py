import torch
import numpy as np
import matplotlib.pyplot as plt

# Parámetros
dz = 1  # Espaciamiento espacial (en metros)
T_max = 200  # Número de pasos temporales
num_points = 50  # Número de puntos espaciales
z_min, z_max = 0, 50  # Límites de la pila de lixiviación en metros (z va de 0 a 50 m)
z_discreto = np.linspace(z_min, z_max, num_points)
# Inicialización de theta[i, 0] (condición inicial)
theta = torch.zeros((num_points, T_max), dtype=torch.float32)
#theta[:, 0] = torch.linspace(0.2, 0.5, num_points)  # Ejemplo de condición inicial no normalizada
theta[:, 0] = torch.tensor(0.01 + 0.39 * np.exp(-0.1 * z_discreto), dtype=torch.float32)
# Flujo externo R[j] constante (por ejemplo, 0.05 m/h)
R = torch.full((T_max,), 0.05, dtype=torch.float32)

# Funciones para K(θ) y D(θ) sin normalización
def K(theta):
    """Conductividad hidráulica K(θ)"""
    return 0.15 * torch.exp(-0.01 * theta)  # Ejemplo basado en valores típicos

def D(theta):
    """Difusión hidráulica D(θ)"""
    return 0.05 + 0.01 * torch.sin(0.1 * theta)  # Ejemplo basado en valores típicos

# Calcular theta[0, j] a lo largo del tiempo
for j in range(1, T_max):
    # Obtener valores actuales de theta[1, j] y theta[2, j]
    theta_1j = theta[1, j - 1]
    theta_2j = theta[2, j - 1]

    # Calcular K(theta[1, j]) y D(theta[1, j])
    K_1j = K(theta_1j)
    D_1j = D(theta_1j)

    # Calcular la condición de borde theta[0, j]
    theta[0, j] = theta_2j + (2 * dz * (R[j] - K_1j) / D_1j)

    # Limitar theta[0, j] a valores válidos si es necesario
    theta[0, j] = torch.clamp(theta[0, j], min=0.0, max=1.0)

# Graficar la evolución de theta[0, j] a lo largo del tiempo
plt.figure(figsize=(10, 6))
plt.plot(range(T_max), theta[0, :].numpy(), label=r"$\theta[0, j]$ (z=0)")
plt.xlabel("Paso temporal (j)")
plt.ylabel(r"$\theta[0, j]$")
plt.title("Evolución de la condición de borde θ[0, j] a lo largo del tiempo")
plt.grid()
plt.legend()
plt.show()
