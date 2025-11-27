import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons

rect_width = 0.5
rect_height = 0.5

# Optimizing Spiral
def optimized_spiral(x_center, y_center, R, max_points, a=0.1):
    total_area = np.pi * R**2  # Area of the circle
    avg_spacing = np.sqrt(total_area / max_points)  # Average spacing between points
    theta_max = R / avg_spacing * 2 * np.pi  # Maximum angle needed
    b = (R - a) / theta_max  # Adjust parameter b to reach radius R
    
    points = []
    theta_values = np.linspace(0, theta_max, max_points)
    for theta in theta_values:
        r = a + b * theta
        if r > R:
            break
        x = x_center + r * np.cos(theta)
        y = y_center + r * np.sin(theta)
        points.append((x, y))
    return points

# Checking Overlapping Frames
def is_overlap(rect1, rect2, max_overlap=0.5):
    x1_min, y1_min = rect1[0] - rect1[2] / 2, rect1[1] - rect1[3] / 2
    x1_max, y1_max = rect1[0] + rect1[2] / 2, rect1[1] + rect1[3] / 2
    x2_min, y2_min = rect2[0] - rect2[2] / 2, rect2[1] - rect2[3] / 2
    x2_max, y2_max = rect2[0] + rect2[2] / 2, rect2[1] + rect2[3] / 2

    overlap_x = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
    overlap_y = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
    overlap_area = overlap_x * overlap_y

    area1 = rect1[2] * rect1[3]
    area2 = rect2[2] * rect2[3]
    min_area = min(area1, area2)

    return overlap_area / min_area > max_overlap

# Drawing Camera Frames
def add_rectangle(center, width, height, rectangles, max_overlap=0.5):
    new_rect = (center[0], center[1], width, height)
    for rect in rectangles:
        if is_overlap(new_rect, rect, max_overlap):
            return new_rect, False
    rectangles.append(new_rect)
    return new_rect, True

# Main function to generate spiral points, check validity, and return valid points
def generate_spiral_and_valid_points(x_center, y_center, R=2, max_points=200, rect_width=rect_width, rect_height=rect_height, max_overlap=0.5):
    # Generate spiral points
    spiral_points = optimized_spiral(x_center, y_center, R, max_points)
    
    valid_rectangles = []
    valid_points = []
    invalid_points = []

    # Check each point and add a rectangle if valid
    for point in spiral_points:
        rect, is_valid = add_rectangle(point, rect_width, rect_height, valid_rectangles, max_overlap)
        if is_valid:
            valid_points.append(point)
        else:
            invalid_points.append(point)

    # Return list of valid points
    return valid_points

# Function to draw the circle and valid points
def plot_valid_points(valid_points, rect_width, rect_height, R=1.8, x_center=0, y_center=0):
    # Create figure
    fig, ax = plt.subplots(figsize=(6, 6))
    plt.subplots_adjust(left=0.25, right=0.85)

    # Plot valid points
    x_valid, y_valid = zip(*valid_points) if valid_points else ([], [])
    ax.plot(x_valid, y_valid, 'bo', label='Valid Points')

    # Draw valid rectangles
    valid_rectangles_patches = []
    for rect in valid_rectangles_patches:
        rect_x = rect[0] - rect[2] / 2
        rect_y = rect[1] - rect[3] / 2
        rectangle = plt.Rectangle((rect_x, rect_y), rect[2], rect[3], color='g', fill=False)
        valid_rectangles_patches.append(rectangle)
        ax.add_patch(rectangle)

    # Draw boundary circle
    circle = plt.Circle((x_center, y_center), R, color='black', fill=False, linestyle='--', linewidth=1)
    ax.add_artist(circle)

    # Display plot
    plt.xlabel("X axis")
    plt.ylabel("Y axis")
    plt.title("Spiral with Valid Points and Rectangles")
    ax.legend()
    plt.show()

# Example usage:
# valid_points = generate_spiral_and_valid_points(0, 0)
# plot_valid_points(valid_points, rect_width, rect_height)
