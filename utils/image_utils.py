from os import listdir
from os.path import isfile, join

import numpy as np
from PIL import Image

# supported image types
image_extensions = {'jpg', 'png', 'jpeg'}


def is_image(file):
    """
    Function to check whether file is a image (by checking file extension)
    """
    file_extensions = file.split('.')
    if len(file_extensions) < 2:
        return False
    return file_extensions[1] in image_extensions


def get_images_in_folder(folder_path):
    """
    Returns all supported images in given folder.
    """
    return [join(folder_path, f) for f in listdir(folder_path) if isfile(join(folder_path, f)) and is_image(f)]


def load_images_to_array(folder_path):
    """
    Method to load all images in given folder and to convert them to (numpy) arrays
    """

    # get all images
    images_paths = get_images_in_folder(folder_path)

    # convert images to numpy arrays
    return [np.array(Image.open(path)) for path in images_paths]
