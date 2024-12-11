import math
import numpy as np
import matplotlib.pyplot as plt

# define the function f(x) = e^x * cos(x)
def f(x):
    return np.exp(x) * np.cos(x)

# define the Taylor polynomial T(x) with default Maclaurin expansion
def T(x, x0=0):
    if x0 == 0:  # Maclaurin polynomial
        return 1 + x - (x**3) / 3
    else:  # Taylor polynomial centered at x0
        return -3.07493232064 - 9.79378201807 * (x - x0) \
               - 6.71884969745 * (x - x0)**2 - 1.2146391256 * (x - x0)**3

# define the remainder (error estimate) for the Taylor approximation
def R(x, x0=0):
    if x0 == 0:  # Maclaurin series remainder
        return (-4 * (x**4)) / math.factorial(4)
    else:  # Taylor series remainder centered at x0
        return (12.2997292826 * (x - x0)**4) / math.factorial(4)

# calculate the actual error between the function and the Taylor approximation
def calc_error(x, x0=0):
    return f(x) - T(x, x0)

# example calculations
x_values = [1, 2]


# nicely formatted output function
def display_output(x):
    print(f"\n{'='*40}")
    print(f"Results for x = {x}:")
    print(f"{'-'*40}")
    print(f"f(x):                  {f(x):.6f}")
    print(f"Maclaurin T(x):         {T(x):.6f}")
    print(f"Taylor T(x) at x0=2:    {T(x, 2):.6f}")
    print(f"Maclaurin remainder R(x): {R(x):.6f}")
    print(f"Taylor remainder R(x) at x0=2: {R(x, 2):.6f}")
    print(f"Maclaurin error:        {calc_error(x):.6f}")
    print(f"Taylor error at x0=2:   {calc_error(x, 2):.6f}")
    print(f"{'='*40}\n")

# printing
for x in x_values:
    display_output(x)



# set up x-values from 0 to 2 with steps of 0.1
xs = [x / 10 for x in range(0, 21)]

# compute the values for f(x), Maclaurin T(x), and Taylor T(x) at x0=2
fs = [f(x) for x in xs]
m3s = [T(x) for x in xs]  # Maclaurin polynomial values
t3s = [T(x, 2) for x in xs]  # Taylor polynomial values at x0 = 2

# calculate the approximate and actual errors for Maclaurin and Taylor series
m3_approx_error = [R(x) for x in xs]
m3_calc_error = [calc_error(x) for x in xs]
t3_approx_error = [R(x, 2) for x in xs]
t3_calc_error = [calc_error(x, 2) for x in xs]

# plot f(x), Maclaurin M_3(x), and Taylor T_3(x) curves
curves = {"f(x)": fs, "Maclaurin M3(x)": m3s, "Taylor T3(x) at x0=2": t3s}

plt.figure(figsize=(8, 6))
for curve_name, curve_data in curves.items():
    plt.plot(xs, curve_data, label=curve_name)

# label the axes and add title and grid
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.title("f(x), Maclaurin M3(x), and Taylor T3(x) curves")
plt.legend()
plt.grid(True)

# display the plot
plt.show()





# define the data for f(x), Maclaurin approximation error, and calculated error
curves = {
    "f(x)": fs,
    "Maclaurin M(x) Approx Error": m3_approx_error,
    "Maclaurin M(x) Calculated Error": m3_calc_error
}

# create the plot
plt.figure(figsize=(8, 6))
for curve_name, curve_data in curves.items():
    plt.plot(xs, curve_data, label=curve_name)

# label the axes and set up title, grid, and legend
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.title("f(x), M(x) Approx Error, M(x) Calculated Error")
plt.legend()
plt.grid(True)

# show the plot
plt.show()



# define the data for f(x), Taylor approximation error, and calculated error
curves = {
    "f(x)": fs,
    "Taylor T(x) Approx Error": t3_approx_error,
    "Taylor T(x) Calculated Error": t3_calc_error
}

# create the plot
plt.figure(figsize=(8, 6))
for curve_name, curve_data in curves.items():
    plt.plot(xs, curve_data, label=curve_name)

# label the axes and set up title, grid, and legend
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.title("f(x), T(x) Approx Error, T(x) Calculated Error")
plt.legend()
plt.grid(True)

# show the plot
plt.show()
