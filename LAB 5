import numpy as np

# Define the objective function (Sphere function: sum(x^2))
def sphere(x):
    return np.sum(x**2)

# Grey Wolf Optimizer (GWO)
class GWO:
    def __init__(self, obj_func, dim, pop_size, max_iter, lb, ub):
        self.obj_func = obj_func  # Objective function to minimize
        self.dim = dim            # Number of dimensions
        self.pop_size = pop_size  # Number of wolves in the population
        self.max_iter = max_iter  # Maximum number of iterations
        self.lb = lb              # Lower bound of search space
        self.ub = ub              # Upper bound of search space

        # Initialize the wolves' positions randomly within bounds
        self.position = np.random.uniform(self.lb, self.ub, (self.pop_size, self.dim))
        self.fitness = np.array([self.obj_func(ind) for ind in self.position])  # Initial fitness of all wolves

        # Initialize the alpha, beta, and delta wolves' positions and fitness
        self.alpha_pos = np.zeros(self.dim)
        self.beta_pos = np.zeros(self.dim)
        self.delta_pos = np.zeros(self.dim)
        self.alpha_score = float('inf')  # Best score (we minimize, so start with infinity)
        self.beta_score = float('inf')
        self.delta_score = float('inf')

    def update_position(self, alpha, beta, delta, a, A, C, position):
        # Update the position of a single wolf based on the positions of alpha, beta, delta
        r1 = np.random.random(self.dim)
        r2 = np.random.random(self.dim)

        # Update position using the equation
        D_alpha = abs(C[0] * r1 - position - alpha)
        D_beta = abs(C[1] * r1 - position - beta)
        D_delta = abs(C[2] * r1 - position - delta)

        X1 = alpha - A[0] * D_alpha
        X2 = beta - A[1] * D_beta
        X3 = delta - A[2] * D_delta

        # New position is the average of the three components
        new_position = (X1 + X2 + X3) / 3
        return new_position

    def optimize(self):
        for t in range(self.max_iter):
            # Update parameters A and C based on the iteration
            a = 2 - t * (2 / self.max_iter)  # Declining over iterations
            A = np.random.uniform(-a, a, 3)
            C = np.random.uniform(0, 2, 3)

            # Evaluate fitness and update the alpha, beta, delta wolves
            for i in range(self.pop_size):
                fitness = self.obj_func(self.position[i])

                if fitness < self.alpha_score:
                    self.alpha_score = fitness
                    self.alpha_pos = self.position[i]

                elif fitness < self.beta_score:
                    self.beta_score = fitness
                    self.beta_pos = self.position[i]

                elif fitness < self.delta_score:
                    self.delta_score = fitness
                    self.delta_pos = self.position[i]

            # Update the position of each wolf in the population
            for i in range(self.pop_size):
                # Update the position of wolf i
                self.position[i] = self.update_position(self.alpha_pos, self.beta_pos, self.delta_pos, a, A, C, self.position[i])

                # Ensure the new position stays within the bounds
                self.position[i] = np.clip(self.position[i], self.lb, self.ub)

            # Optionally print the progress
            print(f"Iteration {t+1}/{self.max_iter} - Best Fitness: {self.alpha_score}")

        return self.alpha_pos, self.alpha_score

# Set problem-specific parameters
dim = 10               # Number of dimensions (variables in the function)
pop_size = 50          # Number of wolves in the population
max_iter = 100         # Number of iterations
lb = -5.12             # Lower bound of search space
ub = 5.12              # Upper bound of search space

# Create an instance of the GWO class
gwo = GWO(obj_func=sphere, dim=dim, pop_size=pop_size, max_iter=max_iter, lb=lb, ub=ub)

# Run the optimization
best_pos, best_score = gwo.optimize()

print("\nOptimization Complete!")
print("Best Solution (Position):", best_pos)
print("Best Fitness (Value):", best_score)
