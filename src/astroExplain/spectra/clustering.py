"""Clustering functions for spectral data.
This module contains functions to perform clustering on spectral data,
including the calculation of inertia and silhouette scores for different
numbers of clusters.
It also includes a function to compress explanation weights by averaging
over fixed segments of the wavelength grid.
"""
import time
from typing import Dict

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score



def group_spectra_by_cluster(
    cluster_labels: np.ndarray,
    anomalies_array: np.ndarray,
    weights: np.ndarray
) -> tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    """
    Group spectra and explanation weights by cluster label.

    Parameters
    ----------
    cluster_labels : np.ndarray
        Array of cluster labels for each sample (shape: [n_samples]).
    anomalies_array : np.ndarray
        Array of input spectra or features corresponding to
        the samples (shape: [n_samples, ...]).
    weights : np.ndarray
        Array of explanation weights for each sample
        (shape: [n_samples, ...]).

    Returns
    -------
    spectra_cluster_dict : dict
        Dictionary mapping each cluster label to the corresponding
        subset of spectra.
    weights_cluster_dict : dict
        Dictionary mapping each cluster label to the corresponding
        subset of explanation weights.
    """
    spectra_cluster_dict = {}
    weights_cluster_dict = {}

    unique_cluster_labels = np.unique(cluster_labels)

    for cluster_label in unique_cluster_labels:
        cluster_mask = cluster_labels == cluster_label
        spectra_cluster_dict[cluster_label] = anomalies_array[cluster_mask]
        weights_cluster_dict[cluster_label] = weights[cluster_mask]
        print(f"Cluster: {cluster_label}, N. spectra: {cluster_mask.sum()}")

    return spectra_cluster_dict, weights_cluster_dict

def get_weights_per_segments(
    weights: np.ndarray,
    n_segments: int
) -> np.ndarray:
    """
    Compress explanation weights by averaging over fixed segments.

    Parameters
    ----------
    weights : np.ndarray
        Array of shape (n_samples, n_wavelengths)
        containing the explanation weights.
    n_segments : int
        Number of segments to divide the wavelength grid into.

    Returns
    -------
    weights_per_segment : np.ndarray
        Compressed array of shape (n_samples, n_segments) where each column
        corresponds to the average weight over a segment.
    """

    n_samples, n_wavelengths = weights.shape

    base_size, residual_size = divmod(n_wavelengths, n_segments)
    if residual_size > 0:
        n_segments += 1

    print(f"Base size: {base_size}, Residual size: {residual_size}")
    print(f"New number of segments: {n_segments}")

    # Create empty array
    weights_per_segment = np.empty((n_samples, n_segments))

    for i in range(n_segments):
        if i < n_segments - 1:
            start = i * base_size
            end = (i + 1) * base_size
        else:
            start = base_size * (n_segments - 1)
            end = n_wavelengths

        # Fill by averaging over the segment
        weights_per_segment[:, i] = weights[:, start:end].mean(axis=1)

    return weights_per_segment

def compute_inertias_silhouette(
    X: np.ndarray, n_clusters: int=10
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate the inertia and silhouette score for a range of cluster numbers.
    Parameters
    ----------
    X : np.ndarray
        The data to cluster.
    n_clusters : int
        The maximum number of clusters to consider.
    Returns
    -------
    inertias : np.ndarray
        The inertia values for each number of clusters.
    silhouette_scores : np.ndarray
        The silhouette scores for each number of clusters.
    """

    inertias = []
    silhouette_scores = []
    n_clusters_range = range(2, n_clusters)

    start_time = time.perf_counter()
    # Fit the k-means model
    for n in n_clusters_range:
        kmeans = KMeans(n_clusters=n, random_state=0)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X, kmeans.labels_))
        # print(f"n: {n}, inertia: {kmeans.inertia_}", end="\r")
    finish_time = time.perf_counter()
    print(f"Run time: {finish_time - start_time:.2f} seconds")

    return np.array(inertias), np.array(silhouette_scores)
