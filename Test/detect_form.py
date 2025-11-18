import os

import cv2
import pytesseract
import numpy as np

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


def detect_card_details(image_path):
    """
    Detects the sign (symbols and text) and background color of a card.

    Parameters:
        image_path (str): Path to the card image.

    Returns:
        dict: Contains detected symbols, text, and the background color.
    """
    # Load the image
    image = cv2.imread(image_path)
    cv2.imshow("original image", image)
    if image is None:
        raise FileNotFoundError("Error: Unable to load the image.")

    print("Image loaded successfully.")

    # Resize for consistency (optional)
    image_resized = cv2.resize(image, (200, 300))  # Adjust dimensions as needed

    # Split the image into two regions: symbol section (bottom) and color/background
    height, width, _ = image_resized.shape
    symbol_region = image_resized[int(height * 0.6):, :]  # Bottom 40% of the card
    background_region = image_resized[:int(height * 0.6), :]  # Top 60% of the card

    # Define the regions to extract
    top_right = image_resized[0:int(height * 0.3), int(width * 0.6):]  # Top right region
    bottom_right = image_resized[int(height * 0.7):, int(width * 0.3):]  # Bottom right region
    top_left = image_resized[0:int(height * 0.3), 0:int(width * 0.6)]  # Top left region
    top_left = image_resized[int(height * 0.1):int(height * 0.2), int(width * 0.09):int(width * 0.2)]  # Top left region
    middle_right = image_resized[int(height * 0.3):int(height * 0.7), int(width * 0.6):]  # Middle right region
    middle_color = image_resized[int(height * 0.95):int(height ), int(width * 0.3):int(width * 0.9)]  # Middle right region

    # Optional: Display the regions to verify
    cv2.imshow("Top Right", top_right)
    cv2.imshow("Bottom Right", bottom_right)

    cv2.imshow("Top Left", top_left)
    cv2.imshow("Middle Right", middle_right)
    cv2.imshow("Middle", middle_color)

    text = pytesseract.image_to_string(top_left, config='--psm 7')  # PSM 7 assumes a single line of text
    print(f"Extracted number: {text.strip()}")

    # Path to the reference shapes directory
    reference_dir = "../Documentation/Forms/"

    best_match = None
    best_match_score = 0

    # Compare the cropped top-left shape with reference shapes
    for filename in os.listdir(reference_dir):
        ref_path = os.path.join(reference_dir, filename)
        reference_image = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)

        if reference_image is None:
            continue  # Skip unreadable files

        # Convert cropped image to grayscale
        top_left_gray = cv2.cvtColor(top_right, cv2.COLOR_BGR2GRAY)

        # Template matching
        result = cv2.matchTemplate(top_left_gray, reference_image, cv2.TM_CCOEFF_NORMED)
        _, score, _, _ = cv2.minMaxLoc(result)

        # Find the best match
        if score > best_match_score:
            best_match_score = score
            best_match = filename

    print(f"Best matching shape: {best_match} with score: {best_match_score}")

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # Step 1: Extract text and symbols using OCR
    gray = cv2.cvtColor(symbol_region, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Run OCR to detect text
    detected_text = pytesseract.image_to_string(thresh, config="--psm 6 digits")
    detected_text = detected_text.strip()
    print(f"Detected Text: {detected_text}")

    # Step 2: Detect the dominant background color
    background_color = extract_card_color(middle_color)
    print(f"Detected Background Color: {background_color}")

    # Step 3: Detect shapes (symbols)
    shapes_info = detect_shapes(symbol_region)

    # Return the detected details
    return {
        "text": detected_text,
        "background_color": background_color,
        "shapes": shapes_info
    }


def extract_card_color(image):
    """
    Extracts the dominant color of a card based on the bottom 10 pixels.

    Parameters:
        card_image_path (str): Path to the card image.

    Returns:
        tuple: The average color as (B, G, R).
    """
    # Get image dimensions
    height, width, _ = image.shape

    # Calculate the average color (BGR format)
    blurred_strip = cv2.GaussianBlur(image, (5, 5), 0)
    average_color = blurred_strip.mean(axis=(0, 1))
    average_color = tuple(map(int, average_color))  # Convert to integers
    # cv2.imshow("Card with Bottom Line", card_with_line)
    print(f"Average color (B, G, R): {average_color}")
    return classify_color(average_color)


# def classify_color(bgr_color):
#     """
#     Classifies a given BGR color into one of the predefined categories:
#     'blue', 'green', 'red', 'yellow', or 'grey'.
#
#     Parameters:
#         bgr_color (tuple): The average color as (B, G, R).
#
#     Returns:
#         str: The name of the closest color.
#     """
#     # Predefined colors (BGR format) with realistic shades for each color
#     color_map = {
#         "blue": [(255, 0, 0), (200, 100, 50)],
#         "green": [(0, 255, 0), (100, 200, 100)],
#         "red": [(0, 0, 255), (50, 50, 200), (70, 89, 179)],
#         "yellow": [(0, 255, 255), (50, 200, 200), (80, 173, 195)],
#         "grey": [(128, 128, 128), (100, 100, 100), (150, 150, 150)],
#     }
#
#     # Convert BGR input color to LAB for perceptual color distance
#     bgr_color_lab = cv2.cvtColor(np.uint8([[bgr_color]]), cv2.COLOR_BGR2LAB)[0][0]
#
#     # Convert color_map to LAB
#     color_map_lab = {
#         color_name: [cv2.cvtColor(np.uint8([[shade]]), cv2.COLOR_BGR2LAB)[0][0] for shade in shades]
#         for color_name, shades in color_map.items()
#     }
#
#     # Calculate the closest color based on Euclidean distance in LAB space
#     min_distance = float("inf")
#     closest_color = None
#
#     for color_name, shades_lab in color_map_lab.items():
#         for shade_lab in shades_lab:
#             distance = np.linalg.norm(np.array(bgr_color_lab) - np.array(shade_lab))
#             if distance < min_distance:
#                 min_distance = distance
#                 closest_color = color_name
#
#     return closest_color

def classify_color(bgr_color):
    """
    Classifies a given BGR color into one of the predefined categories.

    Parameters:
        bgr_color (tuple): The average color as (B, G, R).

    Returns:
        str: The name of the closest color.
    """
    color_map = {
        "blue": [(255, 0, 0), (200, 100, 50)],
        "green": [(0, 255, 0), (100, 200, 100)],
        "red": [(0, 0, 255), (50, 50, 200), (70, 89, 179)],
        "yellow": [(0, 255, 255), (50, 200, 200), (80, 173, 195)],
        "grey": [(128, 128, 128), (100, 100, 100), (150, 150, 150)],
    }
    # Calculate the closest color using Euclidean distance in BGR space
    min_distance = float("inf")
    closest_color = None

    for color_name, shades in color_map.items():
        for shade in shades:
            distance = np.linalg.norm(np.array(bgr_color) - np.array(shade))
            if distance < min_distance:
                min_distance = distance
                closest_color = color_name

    return closest_color
def detect_shapes(region):
    """
    Detects simple shapes (rectangles/squares) in the region using contour detection.

    Parameters:
        region (ndarray): Image region to analyze.

    Returns:
        list: Coordinates of detected shapes.
    """
    shapes = []
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
        if len(approx) == 4:  # Quadrilateral
            shapes.append("rectangle/square")
    print(f"Detected Shapes: {len(shapes)} rectangles/squares")
    return shapes


# Example usage
if __name__ == "__main__":
    card_path = ("../Documentation/Cards/card_35.jpg")  # Update the path to the card image
    try:
        result = detect_card_details(card_path)
        print("\nFinal Detected Details:")
        print(result)
    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
