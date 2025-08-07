import os
import shutil
import imageio.v2 as iio
from pathlib import Path


def create_animation(
    filename_prefix: str = "animation",
    duration: int = 2,
    restart_delay: int = 1000,
    remove_temp_images: bool = True,
    video: bool = False,
):
    """Creates a GIF or MP4 animation from snapshots of the algorithmic process (snapshots must exist in `.outputs/temp/`).

    Args:
        - filename_prefix: filename to use for animation.
        - duration: duration of each frame.
        - restart_delay: helper variable to ensure a pause at the end of any GIF animation.
        - remove_temp_images:
            - True: delete the temp snapshots used to create the animation.
            - False: do NOT delete the temp snapshots used to create the animation.
        - video:
            - False: save animation as GIF
            - True: save the animation as MP4 (requires FFmpeg)

    Returns
        - n/a. The animation is not returned but saved to `./outputs/media/` folder.

    """

    # ASSEMBLE LIST OF IMAGES FILENAMES TO ANIMATE
    images = []
    repository_root: Path = Path(__file__).resolve().parent.parent
    output_folder_path = repository_root / "outputs/media"
    temp_folder_path = repository_root / "outputs/temp"

    image_filenames = os.listdir(temp_folder_path)
    image_filenames = [img for img in image_filenames if img.endswith(".png")]

    # APPEND IMAGES TO AN IMAGES ARRAY
    for filename in image_filenames:
        try:
            image = iio.imread(f"{temp_folder_path}/{filename}")
            images.append(image)
        except FileNotFoundError:
            print(f"Error: Image file not found: ./{temp_folder_path}/{filename}")
            return

    # BUILD THE GIF
    if images:
        Path(output_folder_path).mkdir(parents=True, exist_ok=True)
        iter_duration = [duration] * (len(images) - 1) + [restart_delay]
        if video:
            # Important! Videos require FFmpeg (the actual thing, not just the Python wrapper)
            iio.mimsave(
                f"./{output_folder_path}/{filename_prefix}.mp4", images, fps=0.7
            )
        else:
            iio.mimsave(
                f"./{output_folder_path}/{filename_prefix}.gif",
                images,
                duration=iter_duration,
                loop=0,
            )

    # CLEAN UP TEMPORARY IMAGES
    if remove_temp_images:
        if temp_folder_path.exists():
            shutil.rmtree(temp_folder_path)
