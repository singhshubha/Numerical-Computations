import numpy as np
from scipy.interpolate import CubicSpline

# Data for REG(z) and REG'(z)
z_values = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
REG_values = [0, 0.06, 0.18, 0.34, 0.38, 0.35, 0.3, 0.225, 0.12, 0.035, 0]
REG_prime_values = [0.0583333, -0.00583333, 0.155, 0.109167, -0.00333333, -0.04375,
-0.0616667, -0.09375, -0.101667, -0.0445833, -0.005]

# Construct cubic spline with given derivatives
spline = CubicSpline(z_values, REG_values, bc_type=((1, REG_prime_values[0]), (1, REG_prime_values[-1])))

# Function to evaluate spline interpolation at multiple points
def calculate_spline_for_multiple_points(z_points, spline):
    results = [spline(z) for z in z_points]
    return results

# Test with some points in the interval [zmin, zmax]
z_test_points = [1, 2.5, 3.5, 4.5, 5.5, 6, 8]
S_values = calculate_spline_for_multiple_points(z_test_points, spline)

# Output results
for z, S_z in zip(z_test_points, S_values):
    print(f"Spline interpolation at z={z}: S(z) = {S_z}")








# -------------------------------------------------------------------------------------
# Another easy way of doing Spline Interpolation but using a new mdoule.
# The values are matchign with eachother as well.







# Function to construct cubic spline coefficients
def construct_spline_coefficients(z_values, reg_values, reg_prime_values):
    n = len(z_values) - 1
    a = reg_values
    b = reg_prime_values
    c = np.zeros(n)
    d = np.zeros(n)

    # Calculate coefficients c and d for each interval
    for i in range(n):
        h = z_values[i + 1] - z_values[i]
        c[i] = (3 / h**2) * (reg_values[i + 1] - reg_values[i]) - (2 / h) * reg_prime_values[i] - (1 / h) * reg_prime_values[i + 1]
        d[i] = (-2 / h**3) * (reg_values[i + 1] - reg_values[i]) + (1 / h**2) * (reg_prime_values[i + 1] + reg_prime_values[i])
    
    return a, b, c, d

# Function to evaluate the cubic spline at a point
def evaluate_spline(z, z_values, a, b, c, d):
    # Find the interval for z
    for i in range(len(z_values) - 1):
        if z_values[i] <= z <= z_values[i + 1]:
            h = z - z_values[i]
            return a[i] + b[i] * h + c[i] * h**2 + d[i] * h**3
    return None  # Out of bounds

# Function to calculate spline values for multiple points
def calculate_spline_for_multiple_points(z_points, z_values, reg_values, reg_prime_values):
    a, b, c, d = construct_spline_coefficients(z_values, reg_values, reg_prime_values)
    results = [evaluate_spline(z, z_values, a, b, c, d) for z in z_points]
    return results

# Test with some points in the interval [zmin, zmax]
z_test_points = [1, 2.5, 3.5, 4.5, 5.5, 6, 8]
S_values = calculate_spline_for_multiple_points(z_test_points, z_values, REG_values, REG_prime_values)

# Output results
for z, S_z in zip(z_test_points, S_values):
    print(f"Spline interpolation at z={z}: S(z) = {S_z}")
