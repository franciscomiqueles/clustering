from Clust import Clust
import argparse


if __name__ == "__main__":
    clust = Clust("clust")
    clust.get_data()
    clust.set_folder()
    clust.compute_spike_distance()
    clust.get_dendogram_and_tsne()
    clust.save_data()
    clust.get_bias_histogram()
