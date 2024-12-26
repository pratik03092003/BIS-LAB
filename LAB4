import numpy as np
import math

# Sphere Function: f(x) = sum(x_i^2)
def sphere_function(x):
    return np.sum(x**2)

# Lévy Flight function
def levy_flight(Lambda, d):
    # Lévy flight step size based on power-law distribution
    sigma_u = (math.gamma(1 + Lambda) * np.sin(np.pi * Lambda / 2) /
               (math.gamma((1 + Lambda) / 2) * Lambda * 2**((Lambda - 1) / 2)))**(1 / Lambda)
    u = np.random.normal(0, sigma_u, d)
    v = np.random.normal(0, 1, d)
    step = u / np.abs(v)**(1 / Lambda)
    return step

# Initialize the Cuckoo Search Algorithm
def cuckoo_search(func, n_nests, n_dim, max_iter, pa=0.25, alpha=0.01, lambda_levy=1.5):
    # Initialize nests randomly
    nests = np.random.uniform(-5, 5, (n_nests, n_dim))  # Bound the values between -5 and 5 for the Sphere function
    fitness = np.apply_along_axis(func, 1, nests)  # Calculate fitness of each nest

    # Keep track of the best solution found so far
    best_nest = nests[np.argmin(fitness)]
    best_fitness = np.min(fitness)

    for iteration in range(max_iter):
        # Generate new nests via Lévy flights
        new_nests = nests + alpha * levy_flight(lambda_levy, n_dim)
        # Ensure new nests are within bounds
        new_nests = np.clip(new_nests, -5, 5)

        # Evaluate new nests' fitness
        new_fitness = np.apply_along_axis(func, 1, new_nests)

        # Replace worst nests with new ones based on probability of discovery
        for i in range(n_nests):
            if np.random.rand() < pa:  # Discovery probability
                nests[i] = new_nests[i]
                fitness[i] = new_fitness[i]

        # Update the best solution if we find a better one
        if np.min(fitness) < best_fitness:
            best_fitness = np.min(fitness)
            best_nest = nests[np.argmin(fitness)]

        # Output the current iteration's best solution

    return best_nest, best_fitness

# Set algorithm parameters
n_nests = 50          # Number of nests (solutions)
n_dim = 10            # Dimensionality of the problem (number of variables)
max_iter = 100        # Number of iterations
pa = 0.25             # Probability of discovery (abandoning the worst nests)
alpha = 0.01          # Scaling factor for the Lévy flight
lambda_levy = 1.5     # Exponent for Lévy flight distribution

# Run the Cuckoo Search Algorithm to minimize the Sphere Function
best_solution, best_value = cuckoo_search(sphere_function, n_nests, n_dim, max_iter)

# Output the best solution found
print("\nBest Solution Found:", best_solution)
print("Best Fitness Value:", best_value)
