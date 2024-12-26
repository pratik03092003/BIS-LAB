import numpy as np
import random

# Objective function: f(x) = x^2 + 4x + 4
def objective_function(x):
    return x**2 + 4*x + 4

# GEA parameters
population_size = 30         # Number of genetic sequences (solutions)
num_genes = 1                # Number of genes in each sequence (1D optimization in this case)
mutation_rate = 0.05         # Probability of mutation
crossover_rate = 0.7         # Probability of crossover
num_generations = 100        # Number of generations

# Initialize Population: Generate random genetic sequences (chromosomes)
def initialize_population(population_size, num_genes):
    return np.random.uniform(-10, 10, (population_size, num_genes))

# Fitness Evaluation: Evaluate fitness of each sequence
def evaluate_fitness(population):
    return np.array([objective_function(individual[0]) for individual in population])

# Selection: Tournament Selection
def select(population, fitness):
    selected_indices = np.random.choice(len(population), size=2, replace=False)
    return population[selected_indices[np.argmin(fitness[selected_indices])]]

# Crossover: Single-point crossover
def crossover(parent1, parent2):
    if random.random() < crossover_rate:
        # Random crossover point (for 1D, just average)
        return (parent1 + parent2) / 2
    return parent1

# Mutation: Introduce small random changes
def mutate(individual):
    if random.random() < mutation_rate:
        return individual + np.random.uniform(-1, 1)
    return individual

# Gene Expression: Translate genetic sequence to a functional solution
def gene_expression(population):
    return population

# Main GEA Function: Optimization Loop
def gene_expression_algorithm():
    # Initialize population
    population = initialize_population(population_size, num_genes)

    best_solution = None
    best_fitness = float('inf')

    # Track the best solution through generations
    for generation in range(num_generations):
        fitness = evaluate_fitness(population)

        # Track the best solution
        current_best_index = np.argmin(fitness)
        if fitness[current_best_index] < best_fitness:
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

        # Gene expression (translation of genetic sequence to solutions)
        population = gene_expression(population)

        # Output the best solution at regular intervals
        if generation % 10 == 0:
            print(f"Generation {generation}: Best Fitness = {best_fitness}")

    return best_solution, best_fitness

# Run the GEA
best_solution, best_fitness = gene_expression_algorithm()

# Output final best solution
print(f"Final Best Solution: {best_solution}")
print(f"Final Best Fitness: {best_fitness}")
