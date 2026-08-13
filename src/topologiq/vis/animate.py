"""Create animation from a series of images.

Usage:
    Call `create_animation` from a separate script when there is a need
    to create an animation of the algorithmic process.

Notes:
    The `create_animation` function will NOT work if there are no images
    to stitch together, which the function looks for in `./output/temp`.

"""

import os
from pathlib import Path

import imageio.v2 as iio


def create_animation(
    input_dir: Path,
    output_dir: Path,
    filename_prefix: str = "animation",
    duration: int = 1400,
    restart_delay: int = 1000,
    format: str = "GIF",
):
    """Create a GIF or MP4 animation from snapshots of the process.

    This function reads a series of snapshots of the algorithm's progress after each edge
    and produces an animation based on these snapshots. The function requires
    images to exist in `./output/temp/`.

    Args:
        input_dir: Directory where snapshot PNGs are located.
        output_dir: Directory where animation should be saved to.
        filename_prefix: filename to use for animation.
        duration: duration of each frame.
        restart_delay: helper variable to ensure a pause at the end of any GIF animation.
        format:
            GIF: save animation as GIF
            MP4: save the animation as MP4 (requires FFmpeg)

    """

    # Ensure necessary directories exist
    if not Path.exists(input_dir):
        print("Animation called but no files for animation found.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # Assemble image filenames
    images = []
    image_filenames = os.listdir(input_dir)
    image_filenames = sorted([img for img in image_filenames if img.endswith(".png")])

    for filename in image_filenames:
        try:
            image_path = input_dir / filename
            image = iio.imread(image_path)
            images.append(image)
        except FileNotFoundError:
            return

    # Build animation
    if images:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        iter_duration = [duration] * (len(images) - 1) + [restart_delay]

        # Video (requires FFmpeg)
        if format == "MP4":
            output_file_path = Path(output_dir, f"{filename_prefix}.mp4")
            iio.mimsave(output_file_path, images, fps=2)

        # GIF
        else:
            output_file_path = Path(output_dir, f"{filename_prefix}.gif")
            iio.mimsave(
                output_file_path,
                images,
                duration=iter_duration,
                loop=0,
            )
