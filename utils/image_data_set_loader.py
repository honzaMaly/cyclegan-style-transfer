from zipfile import ZipFile

import numpy as np
import tensorflow as tf
from PIL import Image

from utils.image_utils import is_image


def get_zipped_images(file_name, file_url):
    """
    Method returns images present in (downloaded) zipped file
    :param file_name: basename for the download file (for caching)
    :param file_url: url of resource to be download
    :return: list of images as numpy arrays from (downloaded) ZIP
    """

    # only download file if it doesn't exists, then get path to the file
    destination_path = tf.keras.utils.get_file(file_name, origin=file_url)

    # opening the zip file in READ mode to extract all images in zip
    with ZipFile(destination_path, 'r') as zip_file:
        # get all images in zip and convert them to numpy arrays
        return [np.array(Image.open(zip_file.open(file_name))) for file_name in zip_file.namelist()
                if is_image(file_name)]
