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

def get_closest_explanations_to_centroid(
    n_closest: int,
    weights_cluster: Dict[int, np.ndarray],
    centroids: np.ndarray
) -> Dict[int, np.ndarray]:
    """
    Return indices of the n_closest explanation vectors closest
    to each cluster centroid.

    Parameters
    ----------
    n_closest : int
        Number of explanation vectors to retrieve per cluster that are 
        closest to the centroid.
    weights_cluster : dict of int to np.ndarray
        Dictionary mapping cluster labels to arrays of explanation weights. 
        Each value is an array of shape:
        (n_samples_in_cluster, n_features).
    centroids : np.ndarray
        Array of shape (n_clusters, n_features) containing the centroid of 
        each cluster.

    Returns
    -------
    idx_closest_to_centroid : dict of int to np.ndarray
        Dictionary mapping cluster labels to arrays of indices
        (shape (n_closest,)) of the explanation weights closest
        to each centroid.
    """

    idx_closest_to_centroid = {}

    for label, cluster in weights_cluster.items():

        # reshape for broadcasting
        cluster_centroid = centroids[label].reshape(1, -1)
        distances = np.sum((cluster - cluster_centroid) ** 2, axis=1)

        idx_closest = np.argsort(distances)[:n_closest]
        idx_closest_to_centroid[int(label)] = idx_closest

    return idx_closest_to_centroid


def group_spectra_by_cluster(
    cluster_labels: np.ndarray,
    anomalies_array: np.ndarray,
    weights: np.ndarray,
    print_n_clusters: bool = False
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
    print_n_clusters : bool
        If True, print the number of clusters and the number of
        spectra in each cluster.
        Default is False.

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

        if print_n_clusters:
            print(
                f"Cluster: {cluster_label},"
                f"N. spectra: {cluster_mask.sum()}"
            )

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
