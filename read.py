"""
2 9 6
9 10 3 12
8 11 4 10
4 6 11 5
5 3 8"""

import cv2
import numpy as np
import pytesseract
import matplotlib.pyplot as plt

# Helper function to display images
def show_image(title, image, cmap=None):
    plt.figure(figsize=(10, 10))
    plt.title(title)
    plt.imshow(image, cmap=cmap)
    plt.axis('off')
    plt.show()

# Load the image
image_path = './spielaufbau_fursten_und_forscher.jpg'  # Update this to your image path
image = cv2.imread(image_path)
if image is None:
    raise FileNotFoundError(f"Image not found at path: {image_path}")

# Step 1: Preprocessing with median blur
blurred_color = cv2.medianBlur(image, 11)  # Median blur preserves edges better
#show_image("Blurred Colored Image (Median)", cv2.cvtColor(blurred_color, cv2.COLOR_BGR2RGB))

# Step 2: Convert to grayscale
gray = cv2.cvtColor(blurred_color, cv2.COLOR_BGR2GRAY)
#show_image("Grayscale Image", gray, cmap='gray')

# Step 3: Equalize histogram to improve contrast
equalized_gray = cv2.equalizeHist(gray)
#show_image("Equalized Grayscale Image", equalized_gray, cmap='gray')

# Step 4: Adaptive thresholding
adaptive_thresh = cv2.adaptiveThreshold(
    equalized_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
)
####show_image("Adaptive Threshold", adaptive_thresh, cmap='gray')

# Step 5: Apply morphological operations to clean up the image
kernel = np.ones((3, 3), np.uint8)
morphed = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
#show_image("Morphological Transformations", morphed, cmap='gray')

# Step 6: Use the HoughCircles algorithm
circles = cv2.HoughCircles(
    gray,  # Use the original grayscale image for better results
    cv2.HOUGH_GRADIENT,
    dp=1.2,
    minDist=300,
    param1=100,
    param2=30,
    minRadius=80,
    maxRadius=120
)

# Ensure a limit to the number of circles detected
MAX_CIRCLES = 19

# Prepare the output
output_image = image.copy()
detected_numbers = []

# Define the circular area constraints
image_height, image_width = image.shape[:2]
center_x, center_y = image_width // 2, image_height // 2
area_radius = int(image_height // 2 * 0.85)

# Step 7: Filter circles based on the circular area
if circles is not None:
    circles = np.uint16(np.around(circles))
    filtered_circles = []
    output_image = image.copy()
    
    for circle in circles[0, :]:
        circle_x, circle_y, circle_radius = map(int, circle)  
        # Calculate the distance from the center of the circular area
        distance_from_center = np.sqrt((circle_x - center_x) ** 2 + (circle_y - center_y) ** 2)
        # Check if the circle is within the specified area
        if distance_from_center <= area_radius:
            filtered_circles.append(circle)
            # Draw the circle and its center
            cv2.circle(output_image, (circle_x, circle_y), circle_radius, (0, 255, 0), 2)
            cv2.circle(output_image, (circle_x, circle_y), 2, (0, 0, 255), 3)

    # Show the filtered circles
    if filtered_circles:
        
        pass
        #show_image("Filtered Circles in Specified Area", cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
    else:
        print("No circles detected in the specified area.")
else:
    print("No circles detected.")


print(filtered_circles)
if filtered_circles is not None:
    circles = filtered_circles#np.uint16(np.around(filtered_circles))
    #num_circles = min(len(circles[0, :]), MAX_CIRCLES)
    for idx, circle in enumerate(circles):#[0, :num_circles]):
    #for idx, circle in enumerate(detected_circles):
        print(circle)
        x, y, r = circle
        # Draw the circle perimeter
        cv2.circle(output_image, (x, y), r, (0, 255, 0), 2)
        # Draw the circle center
        cv2.circle(output_image, (x, y), 2, (0, 0, 255), 3)

        retry_factors = [0.675, 0.5]  # Define the factors to reduce r in case of failure
        detected = False  # Flag to indicate if a valid number was found

        for factor in retry_factors:
            # Crop the region of interest (ROI)
            adjusted_r = int(r * factor)
            roi = gray[max(0, y - adjusted_r):y + adjusted_r, max(0, x - adjusted_r):x + adjusted_r]
            blurred_roi = cv2.medianBlur(roi, 7)

            # Threshold the ROI
            roi_binary = cv2.adaptiveThreshold(
                blurred_roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )

            # Perform OCR on the cropped circle
            custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789'
            circle_text = pytesseract.image_to_string(blurred_roi, config=custom_config).strip()

            print(f"Factor: {factor}, Detected text: {circle_text}")

            # Check if OCR result is valid
            if circle_text.isdigit():
                detected_numbers.append(circle_text)
                # Annotate the detected number on the original image
                cv2.putText(
                    output_image, 
                    circle_text, 
                    (x - adjusted_r, y - adjusted_r - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    10, 
                    (255, 0, 0), 
                    5
                )
                detected = True
                break  # Exit the retry loop if a valid number is detected

        if not detected:
            print(f"No valid number detected for circle {idx + 1}.")


# Step 5: Display the final output image with circles and detected numbers
#show_image("Detected Circles and Numbers", cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))

# Output the detected numbers as text
print("Detected Numbers:", detected_numbers)

# Step 6: Save the final image
output_image_path = './detected_circles_with_numbers.jpg'
cv2.imwrite(output_image_path, output_image)
print(f"Output image saved to: {output_image_path}")
