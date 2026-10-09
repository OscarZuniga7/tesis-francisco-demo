import numpy as np
import matplotlib.pyplot as plt

# Parámetros de van Genuchten
theta_r = 0.0126
theta_s = 0.4
alpha = 0.01  # cm^-1
n = 1.5
m = 1 - 1/n
Ks = 0.1  # Conductividad hidráulica saturada (m/h)
dz = 1.0  # Espaciamiento espacial (m)
T_max = 200  # Número de pasos temporales
R = np.full(T_max, 0.01)  # Riego constante (m/h)

# Inicialización de theta
num_points = 50
theta = np.zeros((num_points, T_max + 1))
theta[:, 0] = 0.01 + 0.39 * np.exp(-0.1 * np.linspace(0, 50, num_points))  # Condición inicial

# Funciones para van Genuchten
def Se(theta):
    """Saturación efectiva."""
    theta_clamped = np.clip(theta, theta_r + 1e-6, theta_s - 1e-6)
    return (theta_clamped - theta_r) / (theta_s - theta_r)

def K(theta):
    """Conductividad hidráulica."""
    se = Se(theta)
    return Ks * (se ** 0.5) * (1 - (1 - se ** (1/m)) ** m) ** 2

def C(theta):
    """Capacidad específica del suelo."""
    theta_clamped = np.clip(theta, theta_r + 1e-6, theta_s - 1e-6)
    if theta_clamped > theta_r:
        h = -1 / alpha * ((theta_s - theta_r) / (theta_clamped - theta_r)) ** (1/n)
        return n * m * (theta_s - theta_r) * (alpha ** (-1)) * (alpha * abs(h)) ** (n - 1) / (1 + (alpha * abs(h)) ** n) ** (m + 1)
    else:
        return 1e-6  # Evitar NaN

def D(theta):
    """Difusividad."""
    return K(theta) / C(theta)

# Recalcular theta[0, j]
for j in range(1, T_max + 1):  # j empieza en 1 porque j = 0 es condición inicial
    theta[0, j] = theta[2, j - 1] + (2 * dz * (R[j - 1] - K(theta[1, j - 1]))) / D(theta[1, j - 1])

# Graficar theta[0, j] a lo largo del tiempo
plt.figure(figsize=(8, 5))
plt.plot(range(T_max + 1), theta[0, :], label=r"$\theta[0, j]$")
plt.xlabel("Paso temporal (j)")
plt.ylabel(r"$\theta[0, j]$")
plt.title("Evolución de la condición de borde $\\theta[0, j]$")
plt.legend()
plt.grid()
plt.show()
