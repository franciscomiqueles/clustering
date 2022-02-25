from Clust import Clust
import argparse


if __name__ == "__main__":
    clust = Clust("clust")
    clust.read_params()
    clust.set_folder()
    clust.get_data()
    clust.compute_spike_distance()
    clust.get_dendogram()
    clust.get_tsne()
    clust.save_data()
    clust.get_bias_histogram()
