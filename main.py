import sys

import neo
import numpy as np
import matplotlib.pyplot as plt

def main(filenames):
    """TODO: Docstring for main.
    :returns: TODO

    """
    data = [np.load(x) for x in filenames]
    print(np.shape(data))
    print(type(data[0])) # Type of one read data
    print(data[0].shape) # 120 spike trains per list
    print(type(data[0][0])) # neo spike train object
    print(data[0][0].shape) # Spike train array
    print(data[0][0].annotations) # Dict of neo stuff
    # 'neuron_type' to get exc or inh
    # All annotations ['behav.segm.', 'neuron_type', 'unit_id', 'sua']

    # Get excitatory neurons of the first dataset
    # And drop the inds
    exc = get_neuron_type(data[0], 'exc')[0]
    inh = get_neuron_type(data[0], 'inh')[0]
    print(np.shape(exc))
    print(np.shape(inh))

    # Plot a raster
    # y should grow with neuron index
    # x should be x
    plt.scatter(exc[0], np.ones(exc[0].shape))
    plt.show()

def get_neuron_type(data, neuron_type):
    """
        Filters the data for a given neuron_type (exc or inh)
        Example use: data_exc, eIds = get_neuron_type(data1, 'exc')
    """
    ids = []
    for i in xrange(len(data)):
        if data[i].annotations['neuron_type'] == neuron_type:
            ids.append(i)
    return data[ids], ids

if __name__ == "__main__":
    main(sys.argv[1:])
