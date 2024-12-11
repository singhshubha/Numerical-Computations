import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

z_values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130]
REG_values = [47, 49, 36, 34, 31, 37, 35, 33, 33, 31, 30, 30, 33]
REG_prime_values = []  # This will store the computed derivatives

# Function to compute the derivative using the five-point formula
def five_point_derivative(values, idx, step_size):
    """Applies the five-point formula to estimate the derivative at index idx."""
    return (1 / (12 * step_size)) * (values[idx - 2] - 8 * values[idx - 1] + 8 * values[idx + 1] - values[idx + 2])

# Function to compute the derivative at the start using the start-point formula
def start_point_derivative(values, idx, step_size):
    """Uses the start-point formula to approximate the derivative for the first two indices."""
    return (1 / (12 * step_size)) * (
        -25 * values[idx]
        + 48 * values[idx + 1]
        - 36 * values[idx + 2]
        + 16 * values[idx + 3]  
        - 3 * values[idx + 4]
    )

# Function to compute the derivative at the end using the end-point formula
def end_point_derivative(values, idx, step_size):
    """Uses the end-point formula to approximate the derivative for the last two indices."""
    return (1 / (12 * step_size)) * (
        -25 * values[idx]
        + 48 * values[idx - 1]
        - 36 * values[idx - 2]
        + 16 * values[idx - 3]
        - 3 * values[idx - 4]
    )

# Function to derive the set of derivatives for given z values and REG values
def approximate_derivatives(z_values, REG_values):
    """Calculates the derivative values for the given set of z and REG(z), assuming there are 5 or more points."""
    num_points = len(z_values)
    step_size = z_values[1] - z_values[0]
    
    # Compute derivatives at the start points
    REG_prime_values.append(start_point_derivative(REG_values, 0, step_size))
    REG_prime_values.append(start_point_derivative(REG_values, 1, step_size))
    
    # Compute derivatives at midpoint using five-point formula
    for idx in range(2, num_points - 2):
        derivative = five_point_derivative(REG_values, idx, step_size)
        REG_prime_values.append(derivative)
    
    # Compute derivatives at end points
    REG_prime_values.append(end_point_derivative(REG_values, num_points - 2, step_size))
    REG_prime_values.append(end_point_derivative(REG_values, num_points - 1, step_size))
    
    return REG_prime_values

REG_prime_values = approximate_derivatives(z_values, REG_values)

#-----------------------------------------------------------------------------------------------------------------------------------------



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

lagrange_results = [lagrange_manual(x, z_values, REG_values) for x in z_values]
print(lagrange_results)
lagrange_table = pd.DataFrame({
    "z_values": z_values,
    "Lagrange Results": lagrange_results
})

print(lagrange_table)
#-----------------------------------------------------------------------------------------------------------------------------------------




# Updated Hermite Interpolation. We updated this because we want a section that uses divided difference to calculated the REG_prime_values
def hermite_manual(z, z_values, reg_values, reg_prime_values):
    n = len(z_values)
    Q = [[0] * (2 * n) for _ in range(2 * n)]  # Divided difference table
    Z = [0] * (2 * n)  # Doubled nodes

    # Populate Z with doubled nodes and Q with corresponding REG and REG' values
    for i in range(n):
        Z[2 * i] = Z[2 * i + 1] = z_values[i]
        Q[2 * i][0] = Q[2 * i + 1][0] = reg_values[i]
        Q[2 * i + 1][1] = reg_prime_values[i]  # REG' value
        if i > 0:
            Q[2 * i][1] = (Q[2 * i][0] - Q[2 * i - 1][0]) / (Z[2 * i] - Z[2 * i - 1])

    # Compute higher-order divided differences
    for i in range(2, 2 * n):
        for j in range(2, i + 1):
            Q[i][j] = (Q[i][j - 1] - Q[i - 1][j - 1]) / (Z[i] - Z[i - j])

    # Hermite polynomial evaluation at z
    result = Q[0][0]
    product = 1.0
    for i in range(1, 2 * n):
        product *= (z - Z[i - 1])
        result += Q[i][i] * product

    return result

hermite_results = [hermite_manual(x, z_values, REG_values, REG_prime_values) for x in z_values]
print(hermite_results)
hermite_table = pd.DataFrame({
    "z_values": z_values,
    "Hermite Results": hermite_results
})
print(hermite_table)

#-----------------------------------------------------------------------------------------------------------------------------------------




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


cubic_spline_results = [cubic_spline_manual(x, z_values, REG_values) for x in z_values]
print(cubic_spline_results)
cubic_spline_table = pd.DataFrame({
    "z_values": z_values,
    "Cubic spline Results": cubic_spline_results
})

print(cubic_spline_table)
#-----------------------------------------------------------------------------------------------------------------------------------------



z_fine = np.linspace(min(z_values)+10, max(z_values)-10, 500)


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
plt.xlabel('Time intervals')
plt.ylabel('Punches thrown')
plt.title('Interpolations')
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
