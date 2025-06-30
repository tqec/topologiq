import os
import imageio.v2 as iio
from pathlib import Path
from datetime import datetime


def create_animation(
    filename_prefix: str = "animation",
    duration: int = 2,
    restart_delay: int = 1000,
    remove_temp_images: bool = True
):

    # ASSEMBLE LIST OF IMAGES FILENAMES TO ANIMATE
    images = []
    temp_images_folder_path = "./outputs/temp"
    image_filenames = os.listdir(temp_images_folder_path)
    image_filenames = [img for img in image_filenames if img.endswith(".png")]
    
    # APPEND IMAGES TO AN IMAGES ARRAY
    for filename in image_filenames:
        try:
            image = iio.imread(f"{temp_images_folder_path}/{filename}")
            images.append(image)
        except FileNotFoundError:
            print(f"Error: Image file not found: ./{temp_images_folder_path}/{filename}")
            return

    # BUILD THE GIF
    output_folder_path = "./outputs/gif"
    if images:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        Path(output_folder_path).mkdir(parents=True, exist_ok=True)
        iter_duration = [duration] * (len(images) - 1) + [restart_delay]
        iio.mimsave(
            f"./{output_folder_path}/{filename_prefix}-{timestamp}.gif", images, duration=iter_duration, loop=0
        )
        print(f"GIF animation of build process saved to ./{output_folder_path}/{filename_prefix}-{timestamp}.gif")

    # CLEAN UP TEMPORARY IMAGES
    if remove_temp_images:
        for filename in os.listdir(temp_images_folder_path):
            os.remove(f"./{temp_images_folder_path}/{filename}")
        os.rmdir(f"./{temp_images_folder_path}/")
