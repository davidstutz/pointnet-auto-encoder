import os
import h5py
import argparse
import numpy as np
from matplotlib import pyplot as plt
import mpl_toolkits.mplot3d as mplt

def read_hdf5(file, key = 'tensor'):
    """
    Read a tensor, i.e. numpy array, from HDF5.

    :param file: path to file to read
    :type file: str
    :param key: key to read
    :type key: str
    :return: tensor
    :rtype: numpy.ndarray
    """

    assert os.path.exists(file), 'file %s not found' % file

    h5f = h5py.File(file, 'r')

    assert key in h5f.keys(), 'key %s not found in file %s' % (key, file)
    tensor = h5f[key][()]
    h5f.close()

    return tensor

def plot_point_cloud(points, filepath = '', step = 1):
    """
    Plot a point cloud using the given points.

    :param points: N x 3 point matrix
    :type points: numpy.ndarray
    :param filepath: path to file to save plot to; plot is shown if empty
    :type filepath: str
    :param step: take every step-th point only
    :type step: int
    """

    fig = plt.figure()
    ax = fig.add_subplot(111, projection = '3d')

    xx = points[::step, 0]
    yy = points[::step, 1]
    zz = points[::step, 2]

    ax.scatter(xx, yy, zz, c=zz, s=1)

    if filepath:
        plt.savefig(filepath, bbox_inches='tight')
    else:
        plt.show()

def plot_point_clouds(point_clouds, filepath = ''):
    assert len(point_clouds) > 0

    fig = plt.figure()
    ax = fig.add_subplot(111, projection = '3d')

    c = 0
    for points in point_clouds:
        xx = points[:, 0]
        yy = points[:, 1]
        zz = points[:, 2]

        ax.scatter(xx, yy, zz, c = 0)
        c = c + 1

    if filepath:
        plt.savefig(filepath, bbox_inches='tight')
    else:
        plt.show()

def plot_point_cloud_error(point_clouds, filepath = ''):
    assert len(point_clouds) == 2

    points_a = point_clouds[0]
    points_b = point_clouds[1]

    distances = np.zeros((points_a.shape[0], points_b.shape[0]))
    for n in range(points_a.shape[0]):
        points = np.repeat(points_a[n, :].reshape((1, 3)), points_b.shape[0], axis = 0)
        distances[n, :] = np.sum(np.square(points - points_b), axis = 1).transpose()

    min_indices = np.argmin(distances, axis = 1)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for n in range(points_a.shape[0]):
        ax.plot(np.array([points_a[n, 0], points_b[min_indices[n], 0]]),
                np.array([points_a[n, 1], points_b[min_indices[n], 1]]),
                np.array([points_a[n, 2], points_b[min_indices[n], 2]]))

    if filepath:
        plt.savefig(filepath, bbox_inches='tight')
    else:
        plt.show()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Visualize predictions.')
    parser.add_argument('predictions', type=str, help='Prediction HDF5 file.')
    parser.add_argument('target', type=str, help='Target HDF5 file.')

    args = parser.parse_args()
    if not os.path.exists(args.predictions):
        print('Predictions file does not exist.')
        exit(1)
    if not os.path.exists(args.target):
        print('Target file does not exist.')
        exit(1)

    predictions = read_hdf5(args.predictions)
    predictions = np.squeeze(predictions)
    print('Read %s.' % args.predictions)

    targets = read_hdf5(args.target)
    print('Read %s.' % args.target)

    #print(targets.shape, predictions.shape)
    #assert targets.shape[0] == predictions.shape[0]

    for n in range(min(10, predictions.shape[0])):
        prediction_file = str(n) + '_prediction.png'
        plot_point_cloud(predictions[n], prediction_file)
        print('Wrote %s.' % prediction_file)

        target_file = str(n) + '_target.png'
        plot_point_cloud(targets[n], target_file)
        print('Wrote %s.' % target_file)

        error_file = str(n) + '_error.png'
        plot_point_cloud_error([predictions[n], targets[n]], error_file)
        print('Wrote %s.' % error_file)