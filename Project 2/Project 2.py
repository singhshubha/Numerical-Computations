import math
import matplotlib.pyplot as plt

# define the function f(x)
def f(x):
    return 1 / (math.sqrt(x + 1) + math.sqrt(x))

# define the 3rd-degree Taylor polynomial centered at x = 1
def T3(x):
    return (0.414213562373 
            - 0.14644660940672627 * (x - 1) 
            + 0.16161165235168157 * ((x - 1) ** 2) / 2 
            - 0.3087087392637612 * ((x - 1) ** 3) / 6)

# define the estimated error term for the 3rd-degree polynomial
def R3(x):
    return (0.85464 * ((x - 1) ** 4)) / math.factorial(4)

# calculate the difference between f(x) and T3(x)
def t3_calc_error(x):
    return abs(f(x) - T3(x))

# define the 4th-degree Taylor polynomial centered at x = 1
def T4(x):
    return (0.414213562373 
            - 0.14644660940672627 * (x - 1) 
            + 0.16161165235168157 * ((x - 1) ** 2) / 2 
            - 0.3087087392637612 * ((x - 1) ** 3) / 6 
            + 0.8546359240797 * ((x - 1) ** 4) / 24)

# calculate the difference between f(x) and T4(x)
def t4_calc_error(x):
    return abs(f(x) - T4(x))

# define specific x values for evaluation
xs = [0, 0.5, 0.75, 1, 1.5, 2, 15]

# calculate f(x), T3(x), and T4(x) for the given x values
fs = [f(x) for x in xs]
t3s = [T3(x) for x in xs]
t4s = [T4(x) for x in xs]

# calculate errors for T3 and T4
t3_errors = [t3_calc_error(x) for x in xs]
t4_errors = [t4_calc_error(x) for x in xs]

# print the results
print("x values:", xs)
print("f(x):", fs)
print("T3(x):", t3s)
print("T4(x):", t4s)
print("Error T3:", t3_errors)
print("Error T4:", t4_errors)

# redefine x values from 0 to 2 with a step size of 0.1
xs = [x / 10 for x in range(0, 21)]

# recalculate f(x), T3(x), T4(x) for the new x values
fs = [f(x) for x in xs]
t3s = [T3(x) for x in xs]
t4s = [T4(x) for x in xs]

# recalculate errors for the new x values
t3_errors = [t3_calc_error(x) for x in xs]
t4_errors = [t4_calc_error(x) for x in xs]

# plot f(x), T3(x), and T4(x) using matplotlib
plt.figure(figsize=(10, 5))
plt.plot(xs, fs, label='f(x)')
plt.plot(xs, t3s, label='T3(x)')
plt.plot(xs, t4s, label='T4(x)')
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.title("f(x), T3(x), T4(x)")
plt.legend()
plt.grid(True)
plt.show()

# plot errors for T3(x) and T4(x)
plt.figure(figsize=(10, 5))
plt.plot(xs, t3_errors, label='Error T3')
plt.plot(xs, t4_errors, label='Error T4')
plt.xlabel("X-axis")
plt.ylabel("Error")
plt.title("Error of T3(x) and T4(x)")
plt.legend()
plt.grid(True)
plt.show()
