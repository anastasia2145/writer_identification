import os
import sys
import cv2
import random
import itertools
import numpy as np
from collections import Counter
from scipy.spatial.distance import euclidean
from sklearn.cluster import KMeans
from math import ceil

def get_features(path):
    features = []
    names = []
    skip = 0
    for autor in os.listdir(path):
        segments_path = os.path.join(path, autor)
        segment_number += len(segments_path)
        for segment in os.listdir(segments_path):
            segment_path = os.path.join(segments_path, segment)
            image = cv2.imread(segment_path)
            if np.sum(image) == 0:
                skip += 1
                continue
            features.append(get_features(image))
            names.append(segment_path)
    return features, np.array(names)

def get_clusters(most_100, names, kernel_size = 13):
    start = most_100[0]
    clusters = []
    clusters_images = []
    for i in most_100:
        segments_name = np.ravel(names[np.argwhere(clust == i)])
        line = cv2.imread(segments_name[0])
        images = [line]
        class_segments = [get_features(line)]
        for seg_path in segments_name[1:]:
            img = cv2.imread(seg_path)
            class_segments.append(get_features(img))
            line = np.hstack((line, img))
            images.append(img)
        clusters.append(class_segments)
        clusters_images.append(images)
    return clusters, clusters_images
    
def find_closest_feature_vector(features, images):
    dists = np.zeros((len(features), len(features)))
    for (i, j) in itertools.product(range(len(features)), range(len(features))):
            dists[i, j] = euclidean(features[i], features[j])
    min_ind = np.argmin(np.min(dists, axis=1))
    return features[min_ind], images[min_ind]
 
if __name__ == "__main__": 
    path ='C:/Users/Anastasia/Pictures/autors_segments'
    features, names = get_features(path)
    kmeans = KMeans(n_clusters=150, random_state=0).fit(features)
    clust = kmeans.labels_
    max_cluster_size = Counter(clust).most_common(1)[0][1]
    most_100 = [cluster for (cluster, count) in Counter(clust).most_common(200)[10:110]]

    clusters, clusters_images = get_clusters(most_100, names)

    cluster_centres = []
    for clust, images in zip(clusters, clusters_images):
        feature, image = find_closest_feature_vector(clust, images)
        cluster_centres.append(feature)
         
    cluster_centres = np.array(cluster_centres)
    np.save('cluster_centres_kernel_13', cluster_centres)