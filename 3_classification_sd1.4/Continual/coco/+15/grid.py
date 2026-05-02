from PIL import Image
import os

def create_image_grid(image_files, rows, cols, output_filename="grid_image.png"):
    """
    Creates a grid of images from a list of image file paths.

    Args:
        image_files (list): A list of file paths to the images.
        rows (int): The number of rows for the grid.
        cols (int): The number of columns for the grid.
        output_filename (str): The name of the output grid image file.
    """
    if len(image_files) != rows * cols:
        print("Error: The number of images does not match the grid dimensions.")
        return

    # Open all images and find the maximum width and height
    images = [Image.open(f) for f in image_files]
    max_width = max(img.width for img in images)
    max_height = max(img.height for img in images)

    # Create a new blank image with the size of the grid
    grid_width = max_width * cols
    grid_height = max_height * rows
    grid_image = Image.new('RGB', (grid_width, grid_height), (255, 255, 255))

    # Paste each image into the grid
    for i, img in enumerate(images):
        row = i // cols
        col = i % cols
        x_offset = col * max_width
        y_offset = row * max_height
        grid_image.paste(img, (x_offset, y_offset))

    # Save the final grid image
    grid_image.save(output_filename)
    print(f"Grid image saved as {output_filename}")

if __name__ == '__main__':
    # List of image file names from 1.png to 9.png
    image_list = [f"{i}.png" for i in range(1, 10)]

    # Assuming a 3x3 grid for 9 images
    create_image_grid(image_list, rows=3, cols=3)