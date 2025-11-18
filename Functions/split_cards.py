import cv2
import os


def extract_cards(image_path, output_folder, rows=7, cols=10):
    """
    Extracts cards from an image arranged in a grid layout and saves them as separate files.

    Parameters:
    - image_path (str): Path to the input image file containing the grid of cards.
    - output_folder (str): Path to the folder where extracted cards will be saved.
    - rows (int): Number of rows in the grid (default is 7).
    - cols (int): Number of columns in the grid (default is 10).

    Returns:    # List of configurations for extracting cards
    extraction_configs = [
        {
            "input_image_path": "../Documentation/list_of_cards.jpeg",
            "output_folder_path": "../Documentation/Cards",
            "rows": 7,
            "cols": 10,
        },
        {
            "input_image_path": "../Documentation/list_of_cards_sanctuaries.jpeg",
            "output_folder_path": "../Documentation/Sanctuaries",
            "rows": 5,
            "cols": 10,
        },
    ]

    # Process each configuration
    for config in extraction_configs:
        try:
            print(f"\nProcessing: {config['input_image_path']}")
            extract_cards(
                image_path=config["input_image_path"],
                output_folder=config["output_folder_path"],
                rows=config["rows"],
                cols=config["cols"],
            )
        except FileNotFoundError as e:
            print(f"File not found: {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
    - None. Extracted cards are saved as individual image files in the output folder.
    """
    # Load the image
    image = cv2.imread(image_path)

    # Validate if the image is loaded successfully
    if image is None:
        raise FileNotFoundError(f"Error: Unable to load the image at {image_path}")
    print("Image loaded successfully.")

    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Get image dimensions
    height, width, _ = image.shape
    print(f"Image dimensions: {width} x {height}")

    # Calculate the size of each card
    card_width = width // cols
    card_height = height // rows
    print(f"Card size: {card_width} x {card_height}")

    # Initialize card counter
    card_number = 1

    # Loop through the grid and extract cards
    for row in range(rows):
        for col in range(cols):
            # Calculate the coordinates of the current card
            x1 = col * card_width
            y1 = row * card_height
            x2 = x1 + card_width
            y2 = y1 + card_height

            # Crop the card from the image
            card = image[y1:y2, x1:x2]

            # Generate a filename for the card and save it
            card_filename = os.path.join(output_folder, f"card_{card_number:02d}.jpg")
            cv2.imwrite(card_filename, card)

            # Increment card counter
            card_number += 1

    print(f"All cards have been successfully extracted and saved. \n{image_path}")


# Example usage
if __name__ == "__main__":
    # Define input and output paths

    # List of configurations for extracting cards
    extraction_configs = [
        {
            "input_image_path": "../Documentation/list_of_cards.jpeg",
            "output_folder_path": "../Documentation/Cards",
            "rows": 7,
            "cols": 10,
        },
        {
            "input_image_path": "../Documentation/list_of_cards_sanctuaries.jpeg",
            "output_folder_path": "../Documentation/Sanctuaries",
            "rows": 5,
            "cols": 10,
        },
    ]

    # Process each configuration
    for config in extraction_configs:
        try:
            print(f"\nProcessing: {config['input_image_path']}")
            extract_cards(
                image_path=config["input_image_path"],
                output_folder=config["output_folder_path"],
                rows=config["rows"],
                cols=config["cols"],
            )
        except FileNotFoundError as e:
            print(f"File not found: {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
