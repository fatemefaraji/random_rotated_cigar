import numpy as np
from scipy.stats import ortho_group
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def random_rotated_cigar_function(x):
    # Generate a random orthogonal matrix for rotation
    rotation_matrix = ortho_group.rvs(x.shape[0])
    
    # Rotate the input vector
    rotated_x = np.dot(rotation_matrix, x)

    # Compute the value of the Random Rotated Cigar function
    return np.sum(rotated_x[1:]**2) + 1e6 * rotated_x[0]**2

def pso_random_rotated_cigar(num_particles, num_dimensions, max_iterations, lower_bound, upper_bound):
    # PSO parameters
    w = 0.5       # Inertia weight
    c1 = 1        # Cognitive coefficient
    c2 = 2        # Social coefficient

    # Initialize particles' positions and velocities
    particles_position = np.random.uniform(low=lower_bound, high=upper_bound, size=(num_particles, num_dimensions))
    particles_velocity = np.zeros((num_particles, num_dimensions))
    
    # Initialize personal best positions and fitness
    personal_best_positions = particles_position.copy()
    personal_best_fitness = np.zeros(num_particles)
    
    # Initialize global best position and fitness
    global_best_position = None
    global_best_fitness = float('inf')

    # Store particle positions for plotting
    particle_positions_history = []

    # Main loop for PSO iterations
    for iteration in range(max_iterations):
        # Update each particle's position and velocity
        for i in range(num_particles):
            fitness = random_rotated_cigar_function(particles_position[i])

            # Update personal best
            if fitness < personal_best_fitness[i]:
                personal_best_fitness[i] = fitness
                personal_best_positions[i] = particles_position[i]

            # Update global best
            if fitness < global_best_fitness:
                global_best_fitness = fitness
                global_best_position = particles_position[i].copy()

            # Update velocity and position
            particles_velocity[i] = (w * particles_velocity[i]) + \
                (c1 * np.random.rand() * (personal_best_positions[i] - particles_position[i])) + \
                (c2 * np.random.rand() * (global_best_position - particles_position[i]))

            particles_position[i] = particles_position[i] + particles_velocity[i]

        # Store particle positions for plotting
        particle_positions_history.append(particles_position.copy())

    return global_best_position, particle_positions_history

# Example usage for Random Rotated Cigar function
num_particles = 50
num_dimensions = 2  # 2D for visualization
max_iterations = 50
lower_bound = -5
upper_bound = 5

best_solution, particle_positions_history = pso_random_rotated_cigar(
    num_particles, num_dimensions, max_iterations, lower_bound, upper_bound
)

# Plotting the optimization process
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the function surface
x = np.linspace(lower_bound, upper_bound, 100)
y = np.linspace(lower_bound, upper_bound, 100)
X, Y = np.meshgrid(x, y)
Z = random_rotated_cigar_function(np.vstack([X.ravel(), Y.ravel()]))
ax.plot_surface(X, Y, Z.reshape(X.shape), cmap='viridis', alpha=0.5)

# Plot particle positions over iterations
for i, positions in enumerate(particle_positions_history):
    ax.scatter(positions[:, 0], positions[:, 1], [random_rotated_cigar_function(p) for p in positions], color='r', s=10, alpha=i / max_iterations)

# Highlight the best solution
ax.scatter(best_solution[0], best_solution[1], random_rotated_cigar_function(best_solution), color='g', s=50, label='Best Solution')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Function Value')
ax.set_title('PSO Optimization for Random Rotated Cigar Function')

plt.legend()
plt.show()
