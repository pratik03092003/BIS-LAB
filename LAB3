import random
import math
import numpy as np

# Calculate the Euclidean distance between two cities
def distance(city1, city2):
    return math.sqrt((city1[0] - city2[0])**2 + (city1[1] - city2[1])**2)

# Ant Colony Optimization for TSP
class AntColony:
    def __init__(self, cities, num_ants, alpha, beta, rho, iterations):
        self.cities = cities
        self.num_cities = len(cities)
        self.num_ants = num_ants
        self.alpha = alpha   # Influence of pheromone
        self.beta = beta     # Influence of distance
        self.rho = rho       # Pheromone evaporation rate
        self.iterations = iterations
        self.pheromone = np.ones((self.num_cities, self.num_cities))  # Initial pheromone
        self.distances = np.zeros((self.num_cities, self.num_cities))

        # Calculate the distance matrix for all pairs of cities
        for i in range(self.num_cities):
            for j in range(i + 1, self.num_cities):
                self.distances[i][j] = distance(self.cities[i], self.cities[j])
                self.distances[j][i] = self.distances[i][j]

    def probability(self, ant, city, visited):
        """Calculates the probability of moving to a next city."""
        pheromone = self.pheromone[city]
        heuristic = np.array([1.0 / self.distances[city][i] if i not in visited else 0 for i in range(self.num_cities)])
        pheromone_heuristic = pheromone ** self.alpha * heuristic ** self.beta
        pheromone_heuristic[visited] = 0  # Ensure no city is visited twice

        return pheromone_heuristic / pheromone_heuristic.sum()

    def run(self):
        best_distance = float('inf')
        best_tour = None

        # Iterate for a number of iterations
        for _ in range(self.iterations):
            all_tours = []
            all_distances = []

            # Each ant constructs a solution
            for ant in range(self.num_ants):
                visited = [0]  # Start from city 0
                tour = [0]
                total_distance = 0

                # Construct the solution by visiting all cities
                while len(visited) < self.num_cities:
                    city = visited[-1]
                    prob = self.probability(ant, city, visited)
                    next_city = np.random.choice(range(self.num_cities), p=prob)
                    visited.append(next_city)
                    tour.append(next_city)
                    total_distance += self.distances[city][next_city]

                # Add the return to the starting city
                total_distance += self.distances[visited[-1]][visited[0]]

                # Track the best tour and distance
                all_tours.append(tour)
                all_distances.append(total_distance)

                if total_distance < best_distance:
                    best_distance = total_distance
                    best_tour = tour

            # Update pheromone trails
            self.pheromone *= (1 - self.rho)  # Evaporate pheromone
            for ant in range(self.num_ants):
                for i in range(self.num_cities - 1):
                    city1 = all_tours[ant][i]
                    city2 = all_tours[ant][i + 1]
                    self.pheromone[city1][city2] += 1.0 / all_distances[ant]  # Pheromone reinforcement
                    self.pheromone[city2][city1] += 1.0 / all_distances[ant]

        return best_tour, best_distance

# Example usage
if __name__ == "__main__":
    # Define cities as a list of (x, y) coordinates
    cities = [(0, 0), (1, 3), (4, 3), (6, 1), (6, 5), (2, 7), (3, 4), (5, 2)]

    # Set the ACO parameters
    num_ants = 10
    alpha = 1.0    # Pheromone importance
    beta = 2.0     # Heuristic importance
    rho = 0.1      # Pheromone evaporation
    iterations = 100

    # Create and run the ant colony optimizer
    aco = AntColony(cities, num_ants, alpha, beta, rho, iterations)
    best_tour, best_distance = aco.run()

    # Output the best solution found
    print("Best tour:", best_tour)
    print("Best distance:", best_distance)
