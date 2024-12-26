import numpy as np
import matplotlib.pyplot as plt

# Objective function: f(x) = x^2 + 4x + 4
def objective_function(x):
    return x**2 + 4*x + 4

# PSO parameters
num_particles = 30       # Number of particles
dimensions = 1           # Problem dimensionality (1D for this example)
iterations = 100         # Number of iterations
w = 0.5                  # Inertia weight
c1 = 1.5                 # Cognitive coefficient
c2 = 1.5                 # Social coefficient

# Initialize the particles
positions = np.random.uniform(-10, 10, size=(num_particles, dimensions))  # Random positions
velocities = np.random.uniform(-1, 1, size=(num_particles, dimensions))  # Random velocities
personal_best_positions = np.copy(positions)  # Personal best positions
personal_best_scores = np.array([objective_function(p) for p in positions])  # Personal best scores

# Global best (initially the best personal position)
global_best_position = personal_best_positions[np.argmin(personal_best_scores)]
global_best_score = np.min(personal_best_scores)

# PSO Optimization loop
for iteration in range(iterations):
    for i in range(num_particles):
        # Update velocity
        r1 = np.random.rand()
        r2 = np.random.rand()
        velocities[i] = w * velocities[i] + c1 * r1 * (personal_best_positions[i] - positions[i]) + c2 * r2 * (global_best_position - positions[i])

        # Update position
        positions[i] = positions[i] + velocities[i]

        # Evaluate the objective function
        current_score = objective_function(positions[i])

        # Update personal best
        if current_score < personal_best_scores[i]:
            personal_best_scores[i] = current_score
            personal_best_positions[i] = positions[i]

        # Update global best
        if current_score < global_best_score:
            global_best_score = current_score
            global_best_position = positions[i]

    # Optionally print the global best score during the iterations
    if iteration % 10 == 0:
        print(f"Iteration {iteration}: Global Best Score = {global_best_score}")

# Final result
print(f"Final Global Best Position: {global_best_position}")
print(f"Final Global Best Score: {global_best_score}")

# Plotting the results for visualization
x = np.linspace(-10, 10, 400)
y = objective_function(x)

plt.plot(x, y, label="Objective Function: f(x) = x^2 + 4x + 4", color='blue')
plt.scatter(global_best_position, global_best_score, color='red', label=f"Global Best: {global_best_position[0]:.2f}")
plt.legend()
plt.title("Particle Swarm Optimization Result")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.show()
