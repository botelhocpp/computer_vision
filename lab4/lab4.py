import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

def fit_line_ransac(data_points, iterations, distance_threshold):
    optimal_line = (0, 0, 0, 0)
    most_inliers = []

    for _ in range(iterations):
        # Select two random points to calculate the line
        selected_points = data_points[np.random.choice(data_points.shape[0], 2, replace=False)]
        x_vals, y_vals = selected_points[:, 0], selected_points[:, 1]

        # Check for vertical line
        if x_vals[0] == x_vals[1]:
            continue

        # Line coeficients (y = ax + b)
        a = (y_vals[1] - y_vals[0]) / (x_vals[1] - x_vals[0])
        b = y_vals[1] - a * x_vals[1]

        # Distance of points to line
        dist = np.abs(a * data_points[:, 0] - data_points[:, 1] + b) / np.sqrt(a**2 + 1)
        current_inliers = data_points[dist < distance_threshold]

        # Update the best line if the number of inliners is bigger
        if len(current_inliers) > len(most_inliers):
            most_inliers = current_inliers
            optimal_line = (a, b, x_vals, x_vals)

    return optimal_line, most_inliers

# Main code:

image = cv.imread('pontos_ransac.png')
gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

# Corner detection with Harris
harris_response = cv.cornerHarris(gray_image, 2, 3, 0.05)
corner_threshold = 0.01 * harris_response.max()
corner_locations = harris_response > corner_threshold

interest_points = np.column_stack(np.nonzero(corner_locations))

best_fit_line, inlier_points = fit_line_ransac(interest_points, 50, 5)

x_start, y_start = int(best_fit_line[2][0]), int(best_fit_line[3][0])
x_end, y_end = int(best_fit_line[2][1]), int(best_fit_line[3][1])

x_values = np.linspace(0, image.shape[1], 1000)
y_values = best_fit_line[0] * x_values + best_fit_line[1]

# Plot the image and the line
plt.figure(figsize=(10, 10))
plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
plt.plot(x_values, y_values, color='white')

padding = 20
plt.xlim([-padding, image.shape[1] + padding])
plt.ylim([image.shape[0] + padding, -padding])

plt.axis('off')
plt.show()
