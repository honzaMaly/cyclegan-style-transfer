import numpy as np
import tensorflow as tf

# data to be downloaded... use MNIST test database (located at http://yann.lecun.com/exdb/mnist/)
to_download = {'images': 'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
               'labels': 'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz'}


def get_mnist_data_set():
    """
    Method returns samples of '6' and '9' digit images (using MNIST database). If resources are not present,
    they will be downloaded (and cached).

    :return: numpy array of samples of digit 6, numpy array of samples of digit 9
    """

    def get_resource(file_name, url):
        """
        Download (and cache) given resource and convert it to numpy array (given type).

        :param file_name: basename for the download file. It is also used as a flag to indicate what type of data is
        being processed.
        :param url: url of resource to be download
        :return: numpy array with labels or images of digits
        """

        import gzip

        if file_name != 'labels' and file_name != 'images':
            raise Exception("Unrecognized file: {}".format(file_name))

        # only download file if it doesn't exists, then get path to the file
        destination_path = tf.keras.utils.get_file(file_name, origin=url, extract=False)

        # unzip downloaded file and convert to numpy array
        with gzip.open(destination_path, 'rb') as zipped_file:
            if file_name == 'labels':
                return np.frombuffer(zipped_file.read(), np.uint8, offset=8)
            # convert images - add dimension with colors (RGB)
            images = np.frombuffer(zipped_file.read(), np.uint8, offset=16).reshape((-1, 28, 28, 1))
            return np.concatenate([np.zeros((len(images), 28, 28, 2)), images], axis=3)

    # get resources
    data = {key: get_resource(key, url) for key, url in to_download.items()}

    # function to extract images for a digit. Cast to tensorflow data type
    def extract_images_for_digit_formatted(digit):
        return tf.cast(data['images'][np.argwhere(data['labels'] == digit)].reshape(-1, 28, 28, 3), tf.float32)

    # get images of '9' and '6' instances only
    return extract_images_for_digit_formatted(6), extract_images_for_digit_formatted(9)
