import numpy as np
import matplotlib.pyplot as plt
import math  
import pandas as pd 

# Data Setup
time_intervals = np.arange(5, 155, 10)  # Midpoints of each interval
punches = np.array([47, 49, 36, 34, 31, 37, 35, 33, 33, 31, 30, 30, 33, 28, 27])

# Function Definitions
def taylor_approximation(x, x0, n, data, time_intervals):
    x0_index = np.argmin(np.abs(time_intervals - x0))
    x0 = time_intervals[x0_index]  # Corrected x0 to be a valid entry from time_intervals
    
    derivatives = [data[x0_index]]  # f(x0)
    h = 10  # time interval width
    if n >= 1:
        f_prime = (data[x0_index + 1] - data[x0_index - 1]) / (2 * h) if 0 < x0_index < len(data) - 1 else 0
        derivatives.append(f_prime)
    if n >= 2:
        f_double_prime = (data[x0_index + 1] - 2 * data[x0_index] + data[x0_index - 1]) / h**2 if 1 < x0_index < len(data) - 2 else 0
        derivatives.append(f_double_prime)
    while len(derivatives) <= n:
        derivatives.append(0)
    
    return sum([derivatives[i] * (x - x0)**i / math.factorial(i) for i in range(n + 1)])

def lagrange_interpolation(x, points, data, time_intervals):
    indices = [np.argmin(np.abs(time_intervals - point)) for point in points]
    selected_points = time_intervals[indices]
    selected_values = data[indices]
    
    total = 0
    for i in range(len(selected_points)):
        term = selected_values[i]
        for j in range(len(selected_points)):
            if i != j:
                term *= (x - selected_points[j]) / (selected_points[i] - selected_points[j])
        total += term
    return total

# Calculate Taylor and Lagrange approximations
taylor_results = [taylor_approximation(x, 75, 3, punches, time_intervals) for x in time_intervals]
lagrange_results = [lagrange_interpolation(x, [25, 75, 125], punches, time_intervals) for x in time_intervals]

# Calculating errors
taylor_errors = np.abs(punches - taylor_results)
lagrange_errors = np.abs(punches - lagrange_results)

# Numerical Differentiation Function
def numerical_differentiation(data, h):
    derivatives = []
    for i in range(len(data)):
        if i == 0:
            # Forward difference for the first point
            derivative = (data[i + 1] - data[i]) / h
        elif i == len(data) - 1:
            # Backward difference for the last point
            derivative = (data[i] - data[i - 1]) / h
        else:
            # Central difference for middle points
            derivative = (data[i + 1] - data[i - 1]) / (2 * h)
        derivatives.append(derivative)
    return derivatives

# Calculate the numerical derivatives for punch counts
h = 10  # Interval width in seconds
numerical_derivatives = numerical_differentiation(punches, h)

# Lagrange Derivative Approximation (L'(x)) 
def lagrange_derivative(x, points, data, time_intervals):
    indices = [np.argmin(np.abs(time_intervals - point)) for point in points]
    selected_points = time_intervals[indices]
    selected_values = data[indices]

    derivative_total = 0
    for i in range(len(selected_points)):
        term_derivative = 0
        for j in range(len(selected_points)):
            if i != j:
                product = 1
                for k in range(len(selected_points)):
                    if k != i and k != j:
                        product *= (x - selected_points[k]) / (selected_points[i] - selected_points[k])
                term_derivative += product / (selected_points[i] - selected_points[j])
        derivative_total += selected_values[i] * term_derivative
    return derivative_total

lagrange_derivatives = [lagrange_derivative(x, [25, 75, 125], punches, time_intervals) for x in time_intervals]

# Creating a DataFrame to display all results
df = pd.DataFrame({
    'Time (s)': time_intervals,
    'Actual Punches': punches,
    'Taylor Approximation': taylor_results,
    'Taylor Error': taylor_errors,
    'Lagrange Approximation': lagrange_results,
    'Lagrange Error': lagrange_errors,
    'Numerical Derivative (Punches/second)': numerical_derivatives
})

# Display the DataFrame
print(df)

# Plotting approximations against actual data
plt.figure(figsize=(14, 7))
plt.plot(time_intervals, punches, 'o-', label='Actual Data')
plt.plot(time_intervals, taylor_results, 's--', label='Taylor Approximation')
plt.plot(time_intervals, lagrange_results, 'x--', label='Lagrange Approximation')
plt.title('Approximations vs. Actual Data')
plt.xlabel('Time (seconds)')
plt.ylabel('Punches Thrown')
plt.legend()
plt.show()

# Plotting the errors for Taylor and Lagrange approximations
plt.figure(figsize=(12, 6))
plt.plot(time_intervals, taylor_errors, 's--', label='Taylor Error')
plt.plot(time_intervals, lagrange_errors, 'x--', label='Lagrange Error')
plt.title('Error in Approximations')
plt.xlabel('Time (seconds)')
plt.ylabel('Absolute Error')
plt.legend()
plt.show()


# Plotting L'(x) vs Numerical Differentiation
plt.figure(figsize=(12, 6))
plt.plot(time_intervals, numerical_derivatives, 'o-', label='Numerical Derivative')
plt.plot(time_intervals, lagrange_derivatives, 's--', label="L'(x) (Lagrange Derivative)")
plt.title("Comparison of Numerical Derivatives and Lagrange Derivative L'(x)")
plt.xlabel("Time (seconds)")
plt.ylabel("Rate of Change of Punches (Punches/second)")
plt.legend()
plt.show()
