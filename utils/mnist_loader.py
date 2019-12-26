import os
import numpy as np

# data to be downloaded... use MNIST test database (located at http://yann.lecun.com/exdb/mnist/)
to_download = {'images': 'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
               'labels': 'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz'}


def get_mnist_data_set(dataset_folder_path='data/mnist'):
    """
    Method returns samples of '6' and '9' digit images (using MNIST database). If resources are not present,
    they will be downloaded (and cached).

    :param dataset_folder_path: path where data should be downloaded (cached)
    :return: numpy array of samples of digit 6, numpy array of samples of digit 9
    """

    # create folder with data set (if it doesn't exist)
    os.makedirs(dataset_folder_path, exist_ok=True)

    def get_resource(file_name, url):
        """
        Download (and cache) given resource and convert it to numpy array (given type).

        :param file_name: basename for the download file. It is also used as a flag to indicate what type of data is
        being processed.
        :param url: url of resource to be download
        :return: numpy array with labels or images of digits
        """

        from urllib.error import URLError
        from urllib.request import urlretrieve
        import gzip

        if file_name != 'labels' and file_name != 'images':
            raise Exception("Unrecognized file: {}".format(file_name))

        # path to file
        destination_path = os.path.join(dataset_folder_path, file_name)

        # only download file if it doesn't exists
        if not os.path.exists(destination_path):
            try:
                urlretrieve(url, destination_path)
                print("Downloaded '{}' as '{}'.".format(url, file_name))
            except URLError:
                raise RuntimeError('Error downloading resource {}!'.format(url))
        else:
            print("Skipping download of '{}' (as '{}').".format(url, file_name))

        # unzip downloaded file and convert to numpy array
        with gzip.open(destination_path, 'rb') as zipped_file:
            if file_name == 'labels':
                return np.frombuffer(zipped_file.read(), np.uint8, offset=8)
            return np.frombuffer(zipped_file.read(), np.uint8, offset=16).reshape(-1, 28 * 28)

    # get resources
    data = {key: get_resource(key, url) for key, url in to_download.items()}

    # get images of '9' and '6' instances only
    # to visualize image with matplotlib, use code sample in
    # https://stackoverflow.com/questions/48257255/how-to-import-pre-downloaded-mnist-dataset-from-a-specific-directory-or-folder
    return np.squeeze(data['images'][np.argwhere(data['labels'] == 6)]), np.squeeze(
        data['images'][np.argwhere(data['labels'] == 9)])
