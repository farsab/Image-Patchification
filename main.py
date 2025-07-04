import patchify
from PIL import Image
import numpy as np
import os

def run_patchify_demo():
    os.makedirs("images", exist_ok=True)
    img_path = "images/input.jpg"
    if not os.path.exists(img_path):
        raise FileNotFoundError("Please place an image at 'images/input.jpg'.")

    img = Image.open(img_path).resize((224, 224))
    patches = patchify(img, (16, 16))

    print(f"✔ Image split into {len(patches)} patches of shape {patches[0].shape}")
    np.save("images/patches.npy", patches)

if __name__ == "__main__":
    run_patchify_demo()
