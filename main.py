import sys

import neo
import numpy as np

import validation_functions

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
    # print(data[0][0].annotations) # Dict of neo stuff
    # 'neuron_type' to get exc or inh
    # All annotations ['behav.segm.', 'neuron_type', 'unit_id', 'sua']

    validation_functions.plot_raster(data[0])

    # Interspike intervals for one dataset
    # isi_list = [isi(x) for x in data[0]]
    # print(np.shape(isi_list))

    # Example of calculating mean firing rates of all neurons in one
    # measurement
    mfrs = [validation_functions.mean_firing_rate(x) for x in data[0]]
    print(type(mfrs[0]))

    # Bin spike trains
    # binned_spikes = conversion.BinnedSpikeTrain(data[0], num_bins=50)
    # print(type(binned_spikes))
    # print(dir(binned_spikes))
    # print(np.shape(binned_spikes.spike_indices))
    # Do cch


if __name__ == "__main__":
    main(sys.argv[1:])
