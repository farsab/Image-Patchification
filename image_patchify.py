import numpy as np
from PIL import Image
from typing import Tuple

def patchify(image: Image.Image, patch_size: Tuple[int, int]) -> np.ndarray:
    """
    Splits an image into non-overlapping patches.

    Args:
        image (PIL.Image): Input RGB image.
        patch_size (Tuple[int, int]): Size of each patch (height, width).

    Returns:
        np.ndarray: Array of shape (num_patches, patch_height, patch_width, 3)
    """
    img = np.array(image.convert("RGB"))
    h, w, c = img.shape
    ph, pw = patch_size

    assert h % ph == 0 and w % pw == 0, "Image dimensions must be divisible by patch size."

    patches = img.reshape(h // ph, ph, w // pw, pw, c)
    patches = patches.transpose(0, 2, 1, 3, 4)
    return patches.reshape(-1, ph, pw, c)
