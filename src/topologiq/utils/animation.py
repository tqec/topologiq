"""Util facilities for logging and reading stats.

Usage:
    Call `create_animation` from a separate script when there is a need
    to create an animation of the algorithmic process.

Notes:
    The `create_animation` function will NOT work if there are no images
    to stitch together, which the function looks for in `./output/temp`.

"""

import os
import shutil
from pathlib import Path

import imageio.v2 as iio


def create_animation(
    filename_prefix: str = "animation",
    duration: int = 2,
    restart_delay: int = 1000,
    remove_temp_images: bool = True,
    video: bool = False,
):
    """Create a GIF or MP4 animation from snapshots of the process.

    This function reads a series of snapshots of the algorithm's progress after each edge
    and produces an animation based on these snapshots. The function requires
    images to exist in `./output/temp/`.

    Args:
        filename_prefix: filename to use for animation.
        duration: duration of each frame.
        restart_delay: helper variable to ensure a pause at the end of any GIF animation.
        remove_temp_images:
            True: delete the temp snapshots used to create the animation.
            False: do NOT delete the temp snapshots used to create the animation.
        video:
            False: save animation as GIF
            True: save the animation as MP4 (requires FFmpeg)

    """

    # ASSEMBLE LIST OF IMAGES FILENAMES TO ANIMATE
    images = []
    repo_root: Path = Path(__file__).resolve().parent.parent.parent.parent
    output_folder_path = repo_root / "output/media"
    temp_folder_path = repo_root / "output/temp"

    image_filenames = os.listdir(temp_folder_path)
    image_filenames = sorted([img for img in image_filenames if img.endswith(".png")])

    # APPEND IMAGES TO AN IMAGES ARRAY
    for filename in image_filenames:
        try:
            image_path = temp_folder_path / filename
            image = iio.imread(image_path)
            images.append(image)
        except FileNotFoundError:
            return

    # BUILD THE GIF
    if images:
        Path(output_folder_path).mkdir(parents=True, exist_ok=True)

        iter_duration = [duration] * (len(images) - 1) + [restart_delay]
        if video:
            output_file_path = Path(output_folder_path, f"{filename_prefix}.mp4")
            # Important! Videos require FFmpeg (the actual thing, not just the Python wrapper)
            iio.mimsave(output_file_path, images, fps=0.7)
        else:
            output_file_path = Path(output_folder_path, f"{filename_prefix}.gif")
            iio.mimsave(
                output_file_path,
                images,
                duration=iter_duration,
                loop=0,
            )

    # CLEAN UP TEMPORARY IMAGES
    if remove_temp_images:
        if temp_folder_path.exists():
            shutil.rmtree(temp_folder_path)
