import numpy as np
import random

# Objective function: f(x) = x^2 + 4x + 4
def objective_function(x):
    return x**2 + 4*x + 4

# Parameters
grid_size = 10                # Number of cells in the grid (1D grid here for simplicity)
num_iterations = 100         # Number of iterations
neighborhood_radius = 1      # Neighborhood range (cell's neighbors)
mutation_rate = 0.1          # Probability of mutation

# Initialize the grid: random values within a range (-10, 10)
def initialize_grid(grid_size):
    return np.random.uniform(-10, 10, grid_size)

# Fitness evaluation for each cell
def evaluate_fitness(grid):
    return np.array([objective_function(cell) for cell in grid])

# Update the state of each cell based on its neighbors
def update_states(grid, fitness, neighborhood_radius):
    new_grid = np.copy(grid)
    for i in range(grid_size):
        # Get the neighbors (with wraparound at boundaries)
        left = (i - neighborhood_radius) % grid_size
        right = (i + neighborhood_radius) % grid_size

        # Ensure that the indices are valid
        if left <= right:
            neighbors = grid[left:right+1]
            fitness_neighbors = fitness[left:right+1]
        else:
            # Handle wraparound correctly
            neighbors = np.concatenate([grid[left:], grid[:right+1]])
            fitness_neighbors = np.concatenate([fitness[left:], fitness[:right+1]])

        # Update rule: take the average of neighbors if their fitness is better
        best_neighbor = neighbors[np.argmin(fitness_neighbors)]
        # Update rule: Apply smaller mutation
        new_grid[i] = best_neighbor + random.uniform(-mutation_rate / 10, mutation_rate / 10)  # Reduced mutation impact


    return new_grid

# Main Cellular Algorithm Function
def parallel_cellular_algorithm():
    # Initialize grid
    grid = initialize_grid(grid_size)

    best_solution = None
    best_fitness = float('inf')

    # Iterate through generations
    for iteration in range(num_iterations):
        fitness = evaluate_fitness(grid)

        # Track the best solution
        current_best_index = np.argmin(fitness)
        if fitness[current_best_index] < best_fitness:
            best_fitness = fitness[current_best_index]
            best_solution = grid[current_best_index]

        # Update states based on neighbors
        grid = update_states(grid, fitness, neighborhood_radius)

        # Output the best solution at regular intervals
        if iteration % 10 == 0:
            print(f"Iteration {iteration}: Best Fitness = {best_fitness}")

    return best_solution, best_fitness

# Run the Parallel Cellular Algorithm
best_solution, best_fitness = parallel_cellular_algorithm()

# Output the final best solution
print(f"Final Best Solution: {best_solution}")
print(f"Final Best Fitness: {best_fitness}")
