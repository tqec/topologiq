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
    output_folder_pth = repository_root / "outputs/media"
    temp_folder_pth = repository_root / "outputs/temp"

    image_filenames = os.listdir(temp_folder_pth)
    image_filenames = sorted([img for img in image_filenames if img.endswith(".png")])

    # APPEND IMAGES TO AN IMAGES ARRAY
    for filename in image_filenames:
        try:
            image_pth = temp_folder_pth / filename
            image = iio.imread(image_pth)
            images.append(image)
        except FileNotFoundError:
            return

    # BUILD THE GIF
    if images:
        Path(output_folder_pth).mkdir(parents=True, exist_ok=True)
        
        iter_duration = [duration] * (len(images) - 1) + [restart_delay]
        if video:
            output_file_pth = Path(output_folder_pth, f"{filename_prefix}.mp4")
            # Important! Videos require FFmpeg (the actual thing, not just the Python wrapper)
            iio.mimsave(output_file_pth, images, fps=0.7)
        else:
            output_file_pth = Path(output_folder_pth, f"{filename_prefix}.gif")
            iio.mimsave(
                output_file_pth,
                images,
                duration=iter_duration,
                loop=0,
            )

    # CLEAN UP TEMPORARY IMAGES
    if remove_temp_images:
        if temp_folder_pth.exists():
            shutil.rmtree(temp_folder_pth)


if __name__ == "__main__":
    create_animation(filename_prefix="cnots", video=True)
