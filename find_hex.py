import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
from matplotlib.widgets import Button


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


def random_point_adjustment(detected_numbers, num_points=2000, radius=10, max_iterations=1000):
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
            if dist <= 1.2 * overall_avg_distance:
                # Check if point is not too close to any detected field
                min_dist_to_field = min(distance.euclidean(point, field) for field in field_positions)
                if min_dist_to_field >= 0.25 * overall_avg_distance:
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


# Modified function with buttons for navigation
def animate_evolution(history, detected_numbers, image_path):
    # Load the background image
    img = plt.imread(image_path)

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

    # State for the current frame
    current_frame = [0]  # Use a list to allow modification inside nested functions

    # Update function for displaying a specific frame
    def update(frame):
        current_points = history[frame]
        point_scatter.set_offsets(current_points)
        fig.canvas.draw_idle()

    # Button callback functions
    def next_frame(event):
        if current_frame[0] < len(history) - 1:
            current_frame[0] += 1
            update(current_frame[0])

    def last_frame(event):
        if current_frame[0] > 0:
            current_frame[0] -= 1
            update(current_frame[0])

    # Add buttons for navigation
    ax_next = plt.axes([0.85, 0.05, 0.1, 0.075])  # Position for the "Next" button
    btn_next = Button(ax_next, 'Next')
    btn_next.on_clicked(next_frame)

    ax_last = plt.axes([0.7, 0.05, 0.1, 0.075])  # Position for the "Last" button
    btn_last = Button(ax_last, 'Last')
    btn_last.on_clicked(last_frame)

    # Initialize with the first frame
    update(current_frame[0])

    plt.show()


def plot_evolution_with_background(history, detected_numbers, image_path):
    # Load the background image
    img = plt.imread(image_path)

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
    img = plt.imread(image_path)

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


def adjust_points_to_fields(final_points, field_positions, history, angular_tolerance=15):
    distance_factor = 1.75
    new_points = np.empty((0, 2))  # Initialize as an empty 2D NumPy array
    angular_tolerance_rad = np.deg2rad(angular_tolerance)  # Convert angular tolerance to radians

    for field in field_positions:
        print(field)
        # Calculate distances to all points
        distances = [(point, distance.euclidean(point, field)) for point in np.vstack((final_points, new_points))]
        sorted_points = sorted(distances, key=lambda x: x[1])

        # Take the closest 2 points and calculate their mean distance
        if len(sorted_points) >= 2:
            closest_points = sorted_points[:2]
            mean_distance = np.mean([d for _, d in closest_points])

            # Expand to include other points within 140% of the mean distance
            for point, dist in sorted_points[2:]:
                if dist <= distance_factor * mean_distance:
                    closest_points.append((point, dist))
                else:
                    break  # Stop as the list is sorted by distance

            # Calculate the final mean distance of all close points
            mean_distance = np.mean([d for _, d in closest_points])
            history.append(np.array([p for p, _ in closest_points]))

            nl = list()
            i = 0
            for cp in closest_points:
                if i == 6: 
                    break
                p = cp[0]
                # CAREFUL: Y-axis is down, so angles are different
                closest_angle = np.arctan2(-(p[1] - field[1]), p[0] - field[0])
                if closest_angle < 0:
                    closest_angle += 2 * np.pi
                nl.append((closest_angle, cp[0]))
                i += 1
            print(nl)
            sl = sorted(nl, key=lambda x: x[0])
            print('unsorted')
            print(sl)

            if not len(sl) > 2:
                continue
            i = 0
            while True:
                ds = list()
                for j in range(1, len(sl)):
                    res = sl[j][0] - sl[j-1][0]
                    print(f'{sl[j][0]} - {sl[j-1][0]} = {res}')
                    ds.append(res)
                #d10 = np.mod(sl[1][0] - sl[0][0], np.pi)
                #d21 = np.mod(sl[2][0] - sl[1][0], np.pi)
                #d02 = np.mod(sl[0][0] - sl[2][0], np.pi)
                #print(d10 / np.pi * 180, d21 / np.pi * 180, d02 / np.pi * 180)
                # Check if any of the differences exceed 80 degrees (converted to radians)
                al = list()
                for c in range(len(sl)):
                    a = sl[c][0] 
                    al.append(a)
                print(np.array(al))
                print(ds)
                if any(diff > np.deg2rad(80) for diff in ds):#[d10, d21, d02]):
                    # Rearrange the list 'sl'
                    l = len(sl) - 1
                    sl = [sl[l]] + sl[:l]
                else:
                    break
                
                if i == 10:
                    #exit()
                    history.append(np.vstack((np.array([p for _, p in sl]))))
                    print('sus')

                    break
                i += 1

            print('sorted')
            print(sl)

            cms = list()
            for n in range(len(sl) - 1):
                mean_distance = distance.euclidean(sl[n + 1][1], sl[n][1])
                a = np.sqrt(3) / 2 * mean_distance
                m = sl[n + 1][1] + (sl[n][1] - sl[n + 1][1]) * 0.5
                #history.append(np.vstack((sl[n + 1][1], sl[n][1], m)))
                dir = field - m
                norm = dir / np.linalg.norm(dir)
                cms.append(m + norm * a)
            print('cms')
            print(cms)
            stacked = np.vstack(cms)  # Combine arrays into a 2D array
            cm = np.mean(stacked, axis=0)  # Calculate the mean along the columns

            history.append(np.vstack((stacked, np.array([p for _, p in sl]))))
            #history.append([cm, m] + sl)

            # Add 3 new points around the field at 60° increments
            for i in range(6):
                # Find the closest point to the current field
                #closest_point = min(final_points, key=lambda p: distance.euclidean(p, cm))
                # Calculate the angle of the closest point relative to the field
                #closest_angle = np.arctan2(closest_point[1] - cm[1], closest_point[0] - cm[0])
                angle = sl[-1][0] + np.deg2rad(60 * (i + 1)) 
                x_offset = mean_distance * np.cos(-angle)
                y_offset = mean_distance * np.sin(-angle)
                new_point = (cm[0] + x_offset, cm[1] + y_offset)
                #history.append(np.vstack(new_point))

                # Check if a similar point already exists (distance and angular check)
                is_valid = True
                combined_points = np.vstack((final_points, cm, new_points)) if new_points.size > 0 else final_points
                
                # TODO: enhance
                """
                for p in combined_points:  # Include both existing and new points in the check
                    # Check distance
                    if distance.euclidean(cm, p) <= distance_factor * mean_distance:
                        continue

                    # Check angular proximity
                    existing_angle = np.arctan2(p[1] - cm[1], p[0] - cm[0])
                    new_angle = np.arctan2(new_point[1] - cm[1], new_point[0] - cm[0])
                    if abs(existing_angle - new_angle) <= angular_tolerance_rad:
                        is_valid = False
                        break"""

                if is_valid:
                    new_points = np.vstack((new_points, new_point))  # Add new_point to new_points
                    history.append(np.vstack((new_point, np.array([p for p, _ in closest_points]))))
        print()

    # Add new points to the final points list if there are any new points
    if new_points.size > 0:
        final_points = np.vstack((final_points, new_points))

    return final_points, new_points


# Generate point adjustment history
final_points, history, avg_distances = random_point_adjustment(detected_numbers)

# Apply the adjustment to the final points
final_points, new_points = adjust_points_to_fields(final_points, [tuple(map(int, field[1:])) for field in detected_numbers], history)

# Update the history
#history.append(final_points)

# Animate the evolution
animate_evolution(history, detected_numbers, "detected_circles_with_numbers.jpg")#, save_as_gif=False)

# Visualize the updated history
plot_final_with_background(final_points, detected_numbers, "detected_circles_with_numbers.jpg")
