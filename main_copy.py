import cv2
import numpy as np

def capture_photo(camera_index, output_file):
    # Initialize the camera
    cap = cv2.VideoCapture(camera_index)

    # Check if the camera is opened successfully
    if not cap.isOpened():
        print("Unable to open the camera.")
        return

    # Capture a frame from the camera
    ret, frame = cap.read()

    # Check if the frame was captured successfully
    if not ret:
        print("Failed to capture a frame from the camera.")
        return

    # Save the captured frame to a file
    cv2.imwrite(output_file, frame)
    print(f"Photo saved to {output_file}.")

    # Release the camera
    cap.release()

    return output_file

def modify_image(image_path, levels):
    # Open the image
    image = cv2.imread(image_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Adjust the exposure
    exposure_factor = 0.23  # Example factor for exposure adjustment
    adjusted = np.clip((gray - np.mean(gray)) * exposure_factor + np.mean(gray), 0, 255)

    # Adjust the contrast
    contrast = cv2.convertScaleAbs(adjusted, alpha=0.5, beta=50)  # Adjust alpha and beta values as needed

    # Save the modified image
    output_path = "modified_image.jpg"
    cv2.imwrite(output_path, contrast)
    print(f"Modified image saved as {output_path}")
    return output_path

def convert_to_black_and_white(image_path, threshold):
    # Read the image as grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Apply thresholding to obtain a binary image
    _, binary_image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)

    # Save the modified image
    output_path = "modified_image.jpg"
    cv2.imwrite(output_path, binary_image)
    print(f"Modified image saved as {output_path}")
    return output_path

def find_circles(image_path):
    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Apply Gaussian blur to reduce noise (optional)
    blurred = cv2.GaussianBlur(image, (5, 5), 0)

    # Apply Canny edge detection
    edges = cv2.Canny(blurred, 30, 70)  # Adjust the thresholds as needed

    # Find contours in the edge image
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a copy of the image to draw the contours
    image_with_contours = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Iterate over the contours and identify circles
    for contour in contours:
        # Approximate the contour as a polygon
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.008 * perimeter, True)

        # Check if the contour is circular
        if len(approx) >= 5:
            area = cv2.contourArea(approx)
            radius = perimeter / (2 * np.pi)

            circularity = 4 * np.pi * area / (perimeter * perimeter)

            # Adjust these thresholds according to your requirements
            if 0.2 < circularity < 1.8 and 10 < radius < 500:
                # Draw the circle on the original image
                cv2.drawContours(image_with_contours, [approx], 0, (0, 255, 0), 3)

        # Display the image with contours
    cv2.imshow('Detected Circles', image_with_contours)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


output_file = capture_photo(camera_index, output_file)
image_path = modify_image(output_file, 90)
image_path = convert_to_black_and_white(image_path, threshold)
find_circles(image_path)

