import numpy as np
import tensorflow as tf
from PIL import Image
from PIL import ImageFilter
from random import uniform


def normalize(image):
    """
    Normalize the images to [-1, 1]
    :param image: image to be normalized
    :return: normalized image
    """
    image = tf.cast(image, tf.float32)
    image = (image / 127.5) - 1
    return image


def resize(image, size=(256, 256)):
    """
    Resize the image to the given size
    :param image: image to be resized
    :param size: target size
    :return: resized image
    """
    return tf.image.resize(image, size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)


def random_jitter(image, target_size=(256, 256), mirror=False):
    """
    Apply random jitter on given image
    :param image: original image
    :param target_size: target size
    :param mirror: can be new image mirrored (probability of .5)
    :return: new image
    """

    # perform random crop
    if all(target_size[i] <= round(dim * 0.9) for i, dim in enumerate(image.shape[:2])):
        # to preserve quality of the image
        size = [round(dim * 0.9) for dim in image.shape[:2]] + [image.shape[2]]
        new_image = tf.image.random_crop(image, size=size)
    else:
        # resize image a bit and randomly crop image to original size
        new_image = resize(image, size=[round(dim * 1.1) for dim in image.shape[:2]])
        new_image = tf.image.random_crop(new_image, size=image.shape[:])

    # apply mirroring
    if mirror and tf.random.uniform(()) > 0.5:
        new_image = tf.image.flip_left_right(new_image)

    # resize to target size if necessary
    if target_size is not None and any(dim != target_size[index] for index, dim in enumerate(new_image.shape[:2])):
        new_image = resize(new_image, size=target_size)

    return new_image


def blur_image(image, radius=(1.0, 1.1)):
    """
    Blur given image using 'Gaussian Blur'
    :param radius: radius parameter for 'Gaussian Blur'
    :param image: image to be blurred
    :return: blurred
    """
    return np.array(Image.fromarray(np.array(image))
                    .filter(ImageFilter.GaussianBlur(radius=uniform(*radius))))
