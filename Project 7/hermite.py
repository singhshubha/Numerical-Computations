# Data for REG(z) and REG'(z)
z_values = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
REG_values = [0, 0.06, 0.18, 0.34, 0.38, 0.35, 0.3, 0.225, 0.12, 0.035, 0]
REG_prime_values = [0.0583333, -0.00583333, 0.155, 0.109167, -0.00333333, -0.04375, 
                    -0.0616667, -0.09375, -0.101667, -0.0445833, -0.005]

# Hermite interpolation function
def hermite_interpolation(z, z_values, reg_values, reg_prime_values):
    n = len(z_values)
    Q = [[0] * (2 * n) for _ in range(2 * n)]  # Divided difference table
    Z = [0] * (2 * n)  # Doubled nodes

    # Populate Z with doubled nodes and Q with corresponding REG and REG' values
    for i in range(n):
        Z[2 * i] = Z[2 * i + 1] = z_values[i]
        Q[2 * i][0] = Q[2 * i + 1][0] = reg_values[i]
        Q[2 * i + 1][1] = reg_prime_values[i]
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

# Function to calculate Hermite interpolation for multiple points
def calculate_hermite_for_multiple_points(z_points, z_values, reg_values, reg_prime_values):
    results = []
    for z in z_points:
        result = hermite_interpolation(z, z_values, reg_values, reg_prime_values)
        results.append(result)
    return results

# Test with some points in the interval [zmin, zmax]
z_test_points = [1, 2.5, 3.5, 4.5, 5.5, 6, 8]
H_values = calculate_hermite_for_multiple_points(z_test_points, z_values, REG_values, REG_prime_values)

# Output results
for z, H_z in zip(z_test_points, H_values):
    print(f"Hermite interpolation at z={z}: H(z) = {H_z}")
