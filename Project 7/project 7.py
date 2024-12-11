import numpy as np
import matplotlib.pyplot as plt

z_values = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
REG_values = [0, 0.06, 0.18, 0.34, 0.38, 0.35, 0.3, 0.225, 0.12, 0.035, 0]
REG_prime_values = [0.0583333, -0.00583333, 0.155, 0.109167, -0.00333333, -0.04375,
                    -0.0616667, -0.09375, -0.101667, -0.0445833, -0.005]

# Lagrange Interpolation
def lagrange_manual(x, z_values, reg_values):
    result = 0
    n = len(z_values)
    for i in range(n):
        term = reg_values[i]
        for j in range(n):
            if i != j:
                term *= (x - z_values[j]) / (z_values[i] - z_values[j])
        result += term
    return result

# Hermite Interpolation
def hermite_manual(x, z_values, reg_values, reg_prime_values):
    n = len(z_values)
    Q = [[0] * (2 * n) for _ in range(2 * n)]  
    Z = [0] * (2 * n) 
    for i in range(n):
        Z[2 * i] = Z[2 * i + 1] = z_values[i]
        Q[2 * i][0] = Q[2 * i + 1][0] = reg_values[i]
        Q[2 * i + 1][1] = reg_prime_values[i]
        if i > 0:
            Q[2 * i][1] = (Q[2 * i][0] - Q[2 * i - 1][0]) / (Z[2 * i] - Z[2 * i - 1])
    for i in range(2, 2 * n):
        for j in range(2, i + 1):
            Q[i][j] = (Q[i][j - 1] - Q[i - 1][j - 1]) / (Z[i] - Z[i - j])
    # Hermite polynomial evaluation at x
    result = Q[0][0]
    product = 1.0
    for i in range(1, 2 * n):
        product *= (x - Z[i - 1])
        result += Q[i][i] * product
    return result

# Cubic Spline Interpolation (manual implementation)
def cubic_spline_manual(x, z_values, reg_values):
    n = len(z_values)
    h = np.diff(z_values)
    b = np.diff(reg_values) / h
    u = 2 * (h[:-1] + h[1:])
    v = 6 * (b[1:] - b[:-1])
    u = np.insert(u, 0, 0)
    u = np.append(u, 0)
    v = np.insert(v, 0, 0)
    v = np.append(v, 0)

    # Solve for second derivatives
    m = np.zeros_like(z_values)
    for i in range(1, n - 1):
        m[i] = v[i] / u[i]

    # Evaluate spline
    for i in range(n - 1):
        if z_values[i] <= x <= z_values[i + 1]:
            t = x - z_values[i]
            a = reg_values[i]
            b = (reg_values[i + 1] - reg_values[i]) / h[i] - h[i] * (m[i + 1] + 2 * m[i]) / 6
            c = m[i] / 2
            d = (m[i + 1] - m[i]) / (6 * h[i])
            return a + b * t + c * t**2 + d * t**3
    return 0 

# Generate fine-grained z values
z_fine = np.linspace(0, 10, 500)
L_z = np.array([lagrange_manual(x, z_values, REG_values) for x in z_fine])
H_z = np.array([hermite_manual(x, z_values, REG_values, REG_prime_values) for x in z_fine])
S_z = np.array([cubic_spline_manual(x, z_values, REG_values) for x in z_fine])

# Plot the data and interpolations
plt.figure(figsize=(12, 6))
plt.plot(z_values, REG_values, 'o', label='Original Data', color='black')
plt.plot(z_fine, L_z, label='Lagrange Interpolation', linestyle='dashed')
plt.plot(z_fine, H_z, label='Hermite Interpolation', linestyle='dashdot')
plt.plot(z_fine, S_z, label='Cubic Spline Interpolation', linestyle='solid')
plt.legend()
plt.xlabel('z (mm)')
plt.ylabel('REG(z) (1/hr)')
plt.title('Original Data and Interpolations')
plt.grid()
plt.show()

# Plot the differences
plt.figure(figsize=(12, 6))
plt.plot(z_fine, L_z - H_z, label='Lagrange - Hermite', linestyle='dotted')
plt.plot(z_fine, L_z - S_z, label='Lagrange - Cubic Spline', linestyle='dashed')
plt.plot(z_fine, H_z - S_z, label='Hermite - Cubic Spline', linestyle='dashdot')
plt.legend()
plt.xlabel('z (mm)')
plt.ylabel('Difference')
plt.title('Differences Between Interpolations')
plt.grid()
plt.show()
