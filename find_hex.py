import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
from matplotlib.animation import FuncAnimation

def random_point_adjustment(detected_numbers, num_points=1000, radius=10, max_iterations=1000):
    # Extract field positions
    field_positions = [(int(x), int(y)) for _, x, y in detected_numbers]

    # Randomly distribute points
    points = np.random.randint(
        low=0,
        high=max(max(field_positions))*1.3,
        size=(num_points, 2)
    )

    history = [points.copy()]
    avg_distances = []

    for iteration in range(max_iterations):
        new_points = []
        distances = []

        for point in points:
            # Calculate distances to all fields
            dists = [(field, distance.euclidean(point, field)) for field in field_positions]
            closest_fields = sorted(dists, key=lambda x: x[1])[:3]

            # Move point to the centroid of its closest fields
            closest_positions = [field for field, _ in closest_fields]
            new_x, new_y = np.mean(closest_positions, axis=0)
            new_points.append((new_x, new_y))

            # Record distance
            avg_distance = np.mean([d for _, d in closest_fields])
            distances.append(avg_distance)

        # Update points and record history
        points = np.array(new_points)
        history.append(points)

        # Calculate the overall average distance
        overall_avg_distance = np.mean(distances)
        avg_distances.append(overall_avg_distance)

        # Filter out outlier points
        #points = np.array([p for p, d in zip(points, distances) if d <= 1.15 * overall_avg_distance])
        filtered_points = []
        for point, dist in zip(points, distances):
            if dist <= 1.15 * overall_avg_distance:
                # Check if point is not too close to any detected field
                min_dist_to_field = min(distance.euclidean(point, field) for field in field_positions)
                if min_dist_to_field >= 0.2 * overall_avg_distance:
                    filtered_points.append(point)
        points = np.array(filtered_points)

        # Stop if the points stabilize
        if len(history) > 1 and len(history[-1]) == len(history[-2]) and np.allclose(history[-1], history[-2], atol=1e-2):
            break

    # Remove duplicates by merging points within the radius
    final_points = []
    for i, point in enumerate(points):
        if any(np.linalg.norm(point - other_point) < radius for other_point in final_points):
            continue
        final_points.append(point)

    return np.array(final_points), history, avg_distances


def animate_evolution(history, detected_numbers, image_path, interval=2000, save_as_gif=False):
    # Load the background image
    img = imread(image_path)

    # Extract field positions
    field_positions = [(int(x), int(y)) for _, x, y in detected_numbers]

    # Set up the figure
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_title("Catan Points Adjustment Animation")
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")

    # Plot the background image
    ax.imshow(img, extent=[0, img.shape[1], img.shape[0], 0], origin="upper")

    # Initialize the scatter plots
    field_scatter = ax.scatter(*zip(*field_positions), color='blue', s=100, label="Field")
    point_scatter = ax.scatter([], [], color='red', s=20, label="Point")
    ax.legend()

    # Set axis limits
    ax.set_xlim(0, img.shape[1])
    ax.set_ylim(img.shape[0], 0)  # Invert y-axis

    # Update function for animation
    def update(frame):
        current_points = history[frame]
        point_scatter.set_offsets(current_points)
        return point_scatter,

    # Number of frames is equal to the number of states in history
    frames = len(history)

    # Create animation
    ani = FuncAnimation(fig, update, frames=frames, interval=interval, blit=True, repeat=False)

    if save_as_gif:
        from matplotlib.animation import PillowWriter
        ani.save("points_adjustment.gif", writer=PillowWriter(fps=1000 // interval))

    plt.show()

from matplotlib.pyplot import imread

def plot_evolution_with_background(history, detected_numbers, image_path):
    # Load the background image
    img = imread(image_path)

    # Plot the evolution of points
    for i, points in enumerate(history):
        plt.figure(figsize=(8, 8))
        plt.title(f"Iteration {i + 1}")
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")

        # Plot the background image
        plt.imshow(img, extent=[0, img.shape[1], img.shape[0], 0], origin="upper")

        # Plot detected fields
        field_positions = [(int(x), int(y)) for _, x, y in detected_numbers]
        for x, y in field_positions:
            plt.scatter(x, y, color='blue', s=100, label="Field" if 'Field' not in plt.gca().get_legend_handles_labels()[1] else "")

        # Plot current points
        for x, y in points:
            plt.scatter(x, y, color='red', s=20, label="Point" if 'Point' not in plt.gca().get_legend_handles_labels()[1] else "")

        plt.legend()
        plt.grid(False)
        plt.xlim(0, img.shape[1])
        plt.ylim(img.shape[0], 0)  # Invert the y-axis
        plt.show()

# Final points plot function with background
def plot_final_with_background(final_points, detected_numbers, image_path):
    # Load the background image
    img = imread(image_path)

    plt.figure(figsize=(8, 8))
    plt.title("Final Points")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")

    # Plot the background image
    plt.imshow(img, extent=[0, img.shape[1], img.shape[0], 0], origin="upper")

    # Plot detected fields
    field_positions = [(int(x), int(y)) for _, x, y in detected_numbers]
    for x, y in field_positions:
        plt.scatter(x, y, color='blue', s=100, label="Field" if 'Field' not in plt.gca().get_legend_handles_labels()[1] else "")

    # Plot final points
    for x, y in final_points:
        plt.scatter(x, y, color='red', s=20, label="Point" if 'Point' not in plt.gca().get_legend_handles_labels()[1] else "")

    plt.legend()
    plt.grid(False)
    plt.xlim(0, img.shape[1])
    plt.ylim(img.shape[0], 0)  # Invert the y-axis
    plt.show()


# Example detected numbers
detected_numbers = [
    ('6',  np.uint16(1966), np.uint16(2237)),
    ('11', np.uint16(1573), np.uint16(1663)),
    ('3',  np.uint16(2314), np.uint16(2919)),
    ('5',  np.uint16(3322), np.uint16(2288)),
    ('5',  np.uint16(1569), np.uint16(2827)),
    ('4',  np.uint16(1162), np.uint16(2399)),
    ('11', np.uint16(2677), np.uint16(2243)),
    ('8',  np.uint16(1010), np.uint16(1557)),
    ('10', np.uint16(1996), np.uint16(1039)),
    ('3',  np.uint16(2626), np.uint16(1089)),
    ('9',  np.uint16(1291), np.uint16(1060)),
    ('2',  np.uint16(1631), np.uint16(547)),
    ('4',  np.uint16(2975), np.uint16(1634)),
    ('8',  np.uint16(2998), np.uint16(2917)),
    ('12', np.uint16(3316), np.uint16(1094)),
    ('9',  np.uint16(2323), np.uint16(519)),
    ('10', np.uint16(3646), np.uint16(1673)),
    ('6',  np.uint16(2968), np.uint16(543))
]

"""
# Run the adjustment algorithm
final_points, history, avg_distances = random_point_adjustment(detected_numbers)

# Plot the evolution
plot_evolution_with_background(history, detected_numbers, "detected_circles_with_numbers.jpg")

# Display final points
plt.figure(figsize=(8, 8))
plt.title("Final Points")
plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")
plt.axis("equal")

# Plot detected fields
field_positions = [(int(x), int(y)) for _, x, y in detected_numbers]
for x, y in field_positions:
    plt.scatter(x, y, color='blue', s=100, label="Field" if 'Field' not in plt.gca().get_legend_handles_labels()[1] else "")

# Plot final points
for x, y in final_points:
    plt.scatter(x, y, color='red', s=20, label="Point" if 'Point' not in plt.gca().get_legend_handles_labels()[1] else "")

plt.legend()
plt.grid(True)
plt.show()"""

# Generate point adjustment history
final_points, history, avg_distances = random_point_adjustment(detected_numbers)

# Animate the evolution
animate_evolution(history, detected_numbers, "detected_circles_with_numbers.jpg", interval=500, save_as_gif=False)
