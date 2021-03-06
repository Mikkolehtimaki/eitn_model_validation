import os

import numpy as np
import scipy
from scipy.stats import kruskal
import seaborn as sbs
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

def plot_raster(data, sorting=False, max_time=None, title='Raster plot',
                bg_alpha=0.25):
    """
    Plot raster for one measured dataset. Data should be a list of recordings
    (SpikeTrain objects)
    """

    # Get excitatory neurons of the first dataset
    # And drop the inds
    exc = get_neuron_type(data, 'exc')[0]
    inh = get_neuron_type(data, 'inh')[0]
    if sorting:
        exc = sorted(exc, key=len)
        inh = sorted(inh, key=len)
    sorted_neurons = np.concatenate((exc, inh))

    # Plot a raster
    for idx, neuron in enumerate(sorted_neurons):
        color = 'red' if neuron.annotations['neuron_type'] == 'exc' else 'blue'
        plt.scatter(neuron, idx * np.ones(neuron.shape), s=0.1, c=color)

    # Color the background based on movement annotations
    m_segments = data[0].annotations['behav.segm.']['M']
    for s in m_segments:
        plt.axvspan(s[0], s[0]+s[1], facecolor='y', alpha=bg_alpha)

    m_segments = data[0].annotations['behav.segm.']['RS']
    for s in m_segments:
        plt.axvspan(s[0], s[0]+s[1], facecolor='g', alpha=bg_alpha)

    plt.xlabel('Time (s)')
    plt.ylabel('Neuron')
    plt.title(title)

    if max_time:
        plt.xlim((0, max_time))


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

def kw_pairwise(data, title="Kurskal-Wallis H-test"):
    """
    Test data pairwise in all permutations with the Kruskal Wallis test
    :param data: independent measurements, for example list of lists of spike trains
    :param neuron_type: 'exc', 'inh' if specific neuron type is wanted
    """
    kw_statistics = []
    for sample1 in data:
        temp = []
        for sample2 in data:
            s, p = kruskal(sample1, sample2)
            temp.append(p)
        kw_statistics.append(temp)

    sbs.heatmap(kw_statistics, annot=True, cmap='viridis')
    plt.title(title)
    plt.show()

    return kw_statistics

def filter_data(data, neuron_type=None, behavior=None, window_len=0.5):
    """
    Filter the given list of spike trains on neuron type if given and on
    behavior if given
    """
    allowed_types = ['exc', 'inh']
    allowed_behaviors = ['M', 'RS', 'T']
    allowed_transitions = ['to', 'from']
    # Do nothing if nothing asked
    if neuron_type == None and behavior == None:
        return data

    if neuron_type in allowed_types:
        data = get_neuron_type(data, neuron_type)[0]
    # Filtering for behavior times is a bit more verbose
    if behavior in allowed_behaviors:
        # Extract the annotations, they are same for all neurons in the dataset
        filtered = []
        if behavior == 'T':
            # Transitions _to moving_ state, ugly hardcoded mess
            transitions = []
            transitions.extend(data[0].annotations['behav.segm.']['M'])
            # transitions.extend(data[0].annotations['behav.segm.']['RS'])
            for d in data:  # Loop over the spike trains
                f = []  # Create a list where we append spike times as we filter
                        # through the segments
                for segment in transitions:
                    s1 = segment[0]
                    f.append(d[(d >= s1 - window_len) & (d < s1 + window_len)])
                    # s2 = segment[0] + segment[1]
                    # f.append(d[(d >= s2 - window_len) & (d < s2 + window_len)])
                filtered.extend(f)
        else:
            state_times = data[0].annotations['behav.segm.'][behavior]
            for d in data:  # Loop over the spike trains
                f = []  # Create a list where we append spike times as we filter
                        # through the segments
                for segment in state_times:
                    f.append(d[(d >= segment[0]) & (d < segment[0]+segment[1])])
                filtered.extend(f)
        data = filtered
    return data

def get_all_data():
    """
    Function for returning all data in the dataset
    """
    data_folder = 'data'
    data_names = [
        'data1.npy',
        'data2.npy',
        'data3.npy',
        'data4.npy',
        'data5.npy',
        'data6.npy',
    ]
    return [np.load(os.path.join(data_folder, x)) for x in data_names]
