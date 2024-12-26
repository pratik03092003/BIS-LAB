import numpy as np

# Define the objective function
def objective_function(x):
    return x**2

# Initialize parameters
population_size = 100
mutation_rate = 0.01
crossover_rate = 0.7
num_generations = 50
x_min = -10
x_max = 10

# Create initial population
def initialize_population(size):
    return np.random.uniform(x_min, x_max, size)

# Evaluate fitness
def evaluate_fitness(population):
    return objective_function(population)

# Selection (Tournament Selection)
def select(population, fitness):
    selected_indices = np.random.choice(len(poTpulation), size=2, replace=False)
    return population[selected_indices[np.argmax(fitness[selected_indices])]]

# Crossover
def crossover(parent1, parent2):
    if np.random.rand() < crossover_rate:
        return (parent1 + parent2) / 2  # Simple averaging
    return parent1

# Mutation
def mutate(individual):
    if np.random.rand() < mutation_rate:
        return individual + np.random.uniform(-1, 1)  # Random mutation
    return individual

# Genetic Algorithm
def genetic_algorithm():
    population = initialize_population(population_size)
    best_solution = None
    best_fitness = -np.inf

    for generation in range(num_generations):
        fitness = evaluate_fitness(population)

        # Track the best solution
        current_best_index = np.argmax(fitness)
        if fitness[current_best_index] > best_fitness:
            best_fitness = fitness[current_best_index]
            best_solution = population[current_best_index]

        # Create a new population
        new_population = []
        for _ in range(population_size):
            parent1 = select(population, fitness)
            parent2 = select(population, fitness)
            offspring = crossover(parent1, parent2)
            offspring = mutate(offspring)
            new_population.append(offspring)

        population = np.array(new_population)

    return best_solution, best_fitness

# Run the Genetic Algorithm
best_x, best_value = genetic_algorithm()
print(f"Best x: {best_x}, Maximum value of f(x): {best_value}")
