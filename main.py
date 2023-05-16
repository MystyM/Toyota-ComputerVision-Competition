from PIL import ImageOps
from PIL import ImageEnhance, Image, ImageTk
import cv2
import numpy as np
import tkinter as tk
import matplotlib.pyplot as plt

image_path = "Metal_1.jpg"
output_file = "photo.jpg"
camera_index = 0

def capture_photo(camera_index, output_file):
    # Initialize the camera
    cap = cv2.VideoCapture(camera_index)

    # Check if the camera is opened successfully
    if not cap.isOpened():
        print("Unable to open the camera.")
        return

    # Create a GUI window
    window = tk.Tk()
    window.title("Camera Feed")

    # Create a canvas to display the camera feed
    canvas = tk.Canvas(window, width=640, height=480)
    canvas.pack()

    def show_frame():
        # Capture a frame from the camera
        ret, frame = cap.read()

        # Check if the frame was captured successfully
        if ret:
            # Convert the frame to RGB format
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Resize the frame to fit the canvas
            image = Image.fromarray(rgb_frame)
            image = image.resize((640, 480), Image.ANTIALIAS)

            # Display the frame on the canvas
            photo = ImageTk.PhotoImage(image)
            canvas.create_image(0, 0, anchor=tk.NW, image=photo)
            canvas.image = photo

        # Schedule the next frame capture
        window.after(1, show_frame)

    def capture_button_click():
        # Capture a frame from the camera
        ret, frame = cap.read()

        # Check if the frame was captured successfully
        if ret:
            # Save the captured frame to a file
            cv2.imwrite(output_file, frame)
            print(f"Photo saved to {output_file}.")
            cv2.destroyAllWindows()

    # Create a capture button
    capture_button = tk.Button(window, text="Capture", command=capture_button_click)
    capture_button.pack()

    # Start capturing frames
    show_frame()

    # Start the GUI event loop
    window.mainloop()

    # Release the camera
    cap.release()

    return output_file

def modify_image(image_path, levels):
    # Open the image
    image = Image.open(image_path)

    # Convert the image to "P" mode (256-color palette)
    image = image.convert("P", palette=Image.ADAPTIVE, colors=levels)

    # Convert the image back to RGB mode
    image = image.convert("RGB")

    # Calculate the mean pixel value
    pixels = np.array(image)
    mean_value = np.mean(pixels)

    # Adjust the exposure
    exposure_factor = 0.23  # Example factor for exposure adjustment
    adjusted_pixels = np.clip((pixels - mean_value) * exposure_factor + mean_value, 0, 255)
    adjusted_image = Image.fromarray(adjusted_pixels.astype(np.uint8))

    # Adjust the contrast
    enhancer = ImageEnhance.Contrast(adjusted_image)
    contrast_image = enhancer.enhance(-6)

    # Convert the image to grayscale
    # gray_image = contrast_image.convert("L")

    # Invert the image
    # inverted_image = ImageOps.invert(gray_image)

    # Save the modified image
    output_path = "modified_image.jpg"
    # contrast_image.show()
    contrast_image.save(output_path)
    print(f"Modified image saved as {output_path}")
    return output_path

def convert_to_black_and_white(image_path, threshold):
    image = Image.open(image_path)
    grayscale_image = image.convert("L")

    width, height = grayscale_image.size
    pixels = grayscale_image.load()

    for x in range(width):
        for y in range(height):
            if pixels[x, y] < threshold:
                pixels[x, y] = 0  # Black pixel

    # grayscale_image.show()  # Display the modified image
    grayscale_image.save("modified_image.jpg")
    print(f"Modified image saved as {'modified_image.jpg'}")
    # return grayscale_image

def find_circles(image_path):

    # Load the image
    image = cv2.imread(image_path, 0)  # Read the image as grayscale
    photo = cv2.imread('Metal_1.jpg')

    color_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    cv2.imshow('Color Image', color_image)

    # Apply a threshold to obtain a binary image
    _, thresh = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY)

    # Find contours in the binary image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

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
            if 0.2 < circularity < 1.6 and 20 < radius < 300:
                # Draw the circle on the original image
                cv2.drawContours(photo, [approx], 0, (0, 255, 0), 3)

    # Display the image
    cv2.imshow('Detected Circles', photo)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def draw_contours(image, contours):
    # Draw contours on the image
    cv2.drawContours(image, contours, -1, (0, 255, 255), 2)
def white_to_black(image):
    # Load the image
    image = cv2.imread(image)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise (optional)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply Canny edge detection
    edges = cv2.Canny(blurred, 30, 70)  # Adjust the thresholds as needed

    # Find contours in the edge image
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a copy of the image to draw the contours
    image_with_contours = image.copy()

    # Draw the contours on the image
    draw_contours(image_with_contours, contours)

    # Display the original image and the image with contours
    # cv2.imshow('Original Image', image)
    # cv2.imshow('Image with Contours', image_with_contours)

    return image_with_contours
    []
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def convert_contours_to_jpg(input_image_path, output_image_path):
    # Read the input image
    image = cv2.imread(input_image_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply a threshold to obtain a binary image
    ret, threshold = cv2.threshold(gray, 127, 255, 0)

    # Find contours in the binary image
    contours, hierarchy = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw the contours on the original image
    cv2.drawContours(image, contours, -1, (0, 255, 0), 2)

    # Save the image with contours as a JPG file
    cv2.imwrite(output_image_path, image)

    print("Image with contours saved as", output_image_path)

# output_file = capture_photo(camera_index, output_file)

threshold = 240
# ^^ Adjust this value to change the threshold ^^

source = Image.open(image_path)
source.show()
image_path = modify_image(image_path, 90)


# image_path = convert_to_black_and_white(image_path, threshold)
convert_to_black_and_white(image_path, threshold)

# find_circles("modified_image.jpg")
contours = white_to_black(image_path)

input_path = image_path
output_path = image_path

# convert_contours_to_jpg(input_path, output_path)

find_circles(image_path)