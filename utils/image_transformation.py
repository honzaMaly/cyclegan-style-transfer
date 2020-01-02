import tensorflow as tf


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


def random_jitter(image, target_size=(256, 256), resize_before_cropping=0.1, mirror=False):
    """
    Apply random jitter on given image
    :param image: original image
    :param target_size: target size
    :param resize_before_cropping: should be in interval [0, 0.5] - first, the image is resized by
    1 + resize_before_cropping, then it is randomly cropped
    :param mirror: can be new image mirrored (probability of .5)
    :return: new image
    """

    # resize image a bit and randomly crop image to original size
    if 0 < resize_before_cropping <= 0.5:
        new_image = resize(image, size=[round(dim * (1.0 + resize_before_cropping)) for dim in image.shape[:2]])
        new_image = tf.image.random_crop(new_image, size=image.shape[:])
    else:
        new_image = image

    # apply mirroring
    if mirror and tf.random.uniform(()) > 0.5:
        new_image = tf.image.flip_left_right(new_image)

    # resize to target size if necessary
    if target_size is not None and any(dim != target_size[index] for index, dim in enumerate(new_image.shape[:2])):
        new_image = resize(new_image, size=target_size)

    return new_image
