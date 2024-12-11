import numpy as np
import matplotlib.pyplot as plt

z_values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130]
REG_values = [47, 49, 36, 34, 31, 37, 35, 33, 33, 31, 30, 30, 33]
REG_prime_values = [] # Since we  do nto have any derivatives

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

# Updated Hermite Interpolation. We updated this because we want a section that uses divided difference to calculated the REG_prime_values
def hermite_manual(x, z_values, reg_values):
    n = len(z_values)
    Q = [[0] * (2 * n) for _ in range(2 * n)]
    Z = [0] * (2 * n)
    REG_prime_values = []

    # Populate Z and Q matrices
    for i in range(n):
        Z[2 * i] = Z[2 * i + 1] = z_values[i]
        Q[2 * i][0] = Q[2 * i + 1][0] = reg_values[i]
        if i > 0:
            derivative = (reg_values[i] - reg_values[i - 1]) / (z_values[i] - z_values[i - 1])
            REG_prime_values.append(derivative)
        else:
            REG_prime_values.append(0)  

        Q[2 * i + 1][1] = REG_prime_values[i]
        if i > 0:
            Q[2 * i][1] = (Q[2 * i][0] - Q[2 * i - 1][0]) / (Z[2 * i] - Z[2 * i - 1])

    # Calculate divided differences
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


z_fine = np.linspace(min(z_values), max(z_values), 500)
# Making new values for lagrange and hermite. Leaving spline bacause we are using all values. 
n = len(z_values)
selected_indices = np.linspace(0, n - 1, max(3, n // 5), dtype=int)
lagrange_z_values = [z_values[i] for i in selected_indices]
lagrange_REG_values = [REG_values[i] for i in selected_indices]
hermite_z_values = lagrange_z_values
hermite_REG_values = lagrange_REG_values

# Updated: changed the variables used to the new ones created above
L_z = np.array([lagrange_manual(x, lagrange_z_values, lagrange_REG_values) for x in z_fine])
H_z = np.array([hermite_manual(x, hermite_z_values, hermite_REG_values) for x in z_fine])
S_z = np.array([cubic_spline_manual(x, z_values, REG_values) for x in z_fine])

# Plot the data and interpolations
plt.figure(figsize=(12, 6))
plt.plot(z_values, REG_values, 'o', label='Original Data', color='black')
plt.plot(z_fine, L_z, label='Lagrange Interpolation', linestyle='dashed')
plt.plot(z_fine, H_z, label='Hermite Interpolation', linestyle='dashdot')
plt.plot(z_fine, S_z, label='Cubic Spline Interpolation', linestyle='solid')
plt.legend()
plt.xlabel('Time intervals')
plt.ylabel('Punches thrown')
plt.title('Original Data and Interpolations')
plt.grid()
plt.show()

# Plot the differences
plt.figure(figsize=(12, 6))
plt.plot(z_fine, L_z - H_z, label='Lagrange - Hermite', linestyle='dotted')
plt.plot(z_fine, L_z - S_z, label='Lagrange - Cubic Spline', linestyle='dashed')
plt.plot(z_fine, H_z - S_z, label='Hermite - Cubic Spline', linestyle='dashdot')
plt.legend()
plt.xlabel('Intervals')
plt.ylabel('Difference')
plt.title('Differences Between Interpolations')
plt.grid()
plt.show()
