import numpy as np
import matplotlib.pyplot as plt
from math import factorial, exp
from tabulate import tabulate

# Define functions for the approximations and errors

# Function to compute the true function f(x) = e^x
def f(x):
    return np.exp(x)

# Function to compute the Maclaurin series M(x) for a given degree n
def maclaurin(x, n=3):
    return sum((x**i) / factorial(i) for i in range(n + 1))

# Function to compute the Taylor series T(x) centered at x0 for a given degree n
def taylor(x, x0=2, n=3):
    return sum((exp(x0) * ((x - x0)**i)) / factorial(i) for i in range(n + 1))

# Function to compute the Lagrange polynomial L(x) using given points
def lagrange(x, points):
    def l(i, x):
        xi, yi = points[i]
        terms = [(x - xj) / (xi - xj) for j, (xj, yj) in enumerate(points) if j != i]
        return yi * np.prod(terms)

    return sum(l(i, x) for i in range(len(points)))

# Define points for Lagrange interpolation
points = [(0, f(0)), (1, f(1)), (1.5, f(1.5)), (2, f(2))]

# Calculate and display results for requested x values
x_values = [-1, 0, 0.5, 0.75, 1, 1.5, 2, 15]
results = []

for x in x_values:
    fx = f(x)
    mx = maclaurin(x)
    tx = taylor(x, x0=2)
    lx = lagrange(x, points)
    
    # Calculate errors
    error_mx = abs(fx - mx)
    error_tx = abs(fx - tx)
    error_lx = abs(fx - lx)
    
    # Store results in a list
    results.append((x, fx, mx, error_mx, tx, error_tx, lx, error_lx))

# Print results as a table
print("Results Table:")
print("x\tf(x)\tM(x)\tError M(x)\tT(x)\tError T(x)\tL(x)\tError L(x)")
for row in results:
    print("{:.2f}\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}".format(*row))

# Plot the functions over [0, 2]
x_plot = np.linspace(0, 2, 500)
f_plot = f(x_plot)
m_plot = [maclaurin(x) for x in x_plot]
t_plot = [taylor(x, x0=2) for x in x_plot]
l_plot = [lagrange(x, points) for x in x_plot]

plt.figure(figsize=(14, 8))
plt.plot(x_plot, f_plot, label='f(x) = e^x', color='black', linewidth=2, linestyle='-')
plt.plot(x_plot, m_plot, label='Maclaurin M(x)', color='blue', linewidth=1.5, linestyle='--', marker='o', markevery=50)
plt.plot(x_plot, t_plot, label='Taylor T(x)', color='green', linewidth=1.5, linestyle='-.', marker='s', markevery=50)
plt.plot(x_plot, l_plot, label='Lagrange L(x)', color='red', linewidth=1.5, linestyle=':', marker='^', markevery=50)

plt.title('Function Approximations Over [0, 2]', fontsize=16)
plt.xlabel('x', fontsize=14)
plt.ylabel('Function Value', fontsize=14)
plt.legend(loc='upper left', fontsize=12, frameon=True, shadow=True, fancybox=True)
plt.grid(visible=True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# Plot the errors over [0, 2]
error_m = [abs(f(x) - maclaurin(x)) for x in x_plot]
error_t = [abs(f(x) - taylor(x, x0=2)) for x in x_plot]
error_l = [abs(f(x) - lagrange(x, points)) for x in x_plot]

plt.figure(figsize=(14, 8))
plt.plot(x_plot, error_m, label='Error of M(x)', color='blue', linewidth=1.5, linestyle='--')
plt.plot(x_plot, error_t, label='Error of T(x)', color='green', linewidth=1.5, linestyle='-.')
plt.plot(x_plot, error_l, label='Error of L(x)', color='red', linewidth=1.5, linestyle=':')

plt.title('Error Comparisons of Approximations Over [0, 2]', fontsize=16)
plt.xlabel('x', fontsize=14)
plt.ylabel('Absolute Error', fontsize=14)
plt.legend(loc='upper right', fontsize=12, frameon=True, shadow=True, fancybox=True)
plt.grid(visible=True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()


# Define columns and display the results as a table
columns = ["x", "f(x)", "M(x)", "Error M(x)", "T(x)", "Error T(x)", "L(x)", "Error L(x)"]
table = tabulate(results, headers=columns, tablefmt="pretty", floatfmt=".5f")

# Print the table
print("Results Table:")
print(table)

