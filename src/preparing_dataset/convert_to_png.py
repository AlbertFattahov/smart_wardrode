from PIL import Image
import os

folder = "test_images/"

for file in os.listdir(folder):
    if file.lower().endswith(".jfif"):
        os.remove(os.path.join(folder, file))