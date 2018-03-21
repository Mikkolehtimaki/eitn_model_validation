import numpy as np
import matplotlib.pyplot as plt
from elephant.statistics import isi
from elephant.spike_train_correlation import cch
import elephant.conversion as conversion

def mean_firing_rate(spike_train):
    """
    Calculate the mean firing rate per spike train
    :param spike_train: neo.SpikeTrain object

    Returns the value as a float.
    """
    num_spikes = len(spike_train)
    measurement_length = spike_train.t_stop - spike_train.t_start
    return np.divide(num_spikes, measurement_length.magnitude)

def plot_raster(data):
    """
    Plot raster for one measured dataset. Data should be a list of recordings
    (SpikeTrain objects)
    """

    # Get excitatory neurons of the first dataset
    # And drop the inds
    exc = get_neuron_type(data, 'exc')[0]
    inh = get_neuron_type(data, 'inh')[0]
    sorted_neurons = np.concatenate((exc, inh))

    # Plot a raster
    # y should grow with neuron index
    # x should be x
    for idx, neuron in enumerate(sorted_neurons):
        color = 'red' if neuron.annotations['neuron_type'] == 'exc' else 'blue'
        plt.scatter(neuron, idx * np.ones(neuron.shape), s=0.1, c=color)

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
