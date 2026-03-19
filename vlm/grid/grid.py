from PIL import Image
import os
import shutil

def create_image_grid(image_files, rows, cols, output_filename):
    """
    Creates a grid of images from a list of image file paths.

    Args:
        image_files (list): A list of file paths to the images.
        rows (int): The number of rows for the grid.
        cols (int): The number of columns for the grid.
        output_filename (str): The name of the output grid image file.
    """
    if len(image_files) != rows * cols:
        print(f"Error: The number of images ({len(image_files)}) does not match the grid dimensions ({rows}x{cols}).")
        return

    try:
        images = [Image.open(f) for f in image_files]
        max_width = max(img.width for img in images)
        max_height = max(img.height for img in images)

        grid_width = max_width * cols
        grid_height = max_height * rows
        grid_image = Image.new('RGB', (grid_width, grid_height), (255, 255, 255))

        for i, img in enumerate(images):
            row = i // cols
            col = i % cols
            x_offset = col * max_width
            y_offset = row * max_height
            grid_image.paste(img, (x_offset, y_offset))

        grid_image.save(output_filename)
        print(f"Grid image saved as {output_filename}")
    except FileNotFoundError as e:
        print(f"Error: A file was not found. Please check the image paths. Details: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == '__main__':
    target_dirs = [        
        "/mnt/home/yhgil99/unlearning/Soft_manipulation/output_img/CNB/ghost_attn_debug7.5mu0.1"
    ]

    # 각 폴더에 grid.png를 생성하고, 현재 디렉터리에도 별도로 저장
    for i, directory in enumerate(target_dirs):
        # 1. 각 폴더에 grid.png 생성
        output_path_in_dir = os.path.join(directory, "grid.png")
        image_list = [os.path.join(directory, f"{i}.png") for i in range(1, 19)]
        print(f"\nProcessing directory: {directory}")
        create_image_grid(image_list, rows=3, cols=6, output_filename=output_path_in_dir)

        # 2. 현재 디렉터리에도 grid.png를 복사해서 저장
        # os.getcwd()로 현재 실행 디렉터리 경로를 가져옴
        current_dir = os.getcwd()
        # 파일명을 'grid_1.png', 'grid_2.png' 등으로 설정
        output_path_in_current_dir = os.path.join(current_dir, f"gridasdjflksadjklf_{i+1}.png")
        
        # shutil.copy()를 사용하여 파일 복사
        try:
            shutil.copy(output_path_in_dir, output_path_in_current_dir)
            print(f"Copied {os.path.basename(output_path_in_dir)} to {output_path_in_current_dir}")
        except FileNotFoundError:
            print(f"Error: {output_path_in_dir} was not found for copying.")
        except Exception as e:
            print(f"An unexpected error occurred while copying: {e}")