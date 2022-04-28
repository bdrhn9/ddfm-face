#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering
from sklearn_extra.cluster import KMedoids

import torch
import torch.nn.functional as F
from torchvision import transforms as T
from torch.utils.data import DataLoader

from backbones.resnet_vgg import resnet50
from modules.utils import Clustered_Dataset

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

train_transform = T.Compose(
    [
        T.Resize((112, 112)),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)


dataset = Clustered_Dataset(
    folder_path="./data/esogu_faces",
    meta_path="./data/esogu_faces/train_meta.npy",
    transform=train_transform,
    file_path=True,
)

dataloader = DataLoader(
    dataset, batch_size=64, shuffle=False, num_workers=0, pin_memory=True
)


model = IR_50(pretrained=True, input_size=112).cuda()
model.eval()

all_features, all_labels, all_path = [], [], []
with torch.no_grad():
    for batch_id, (inputs, labels, path) in enumerate(dataloader):
        inputs = inputs.cuda()
        labels = labels.numpy()

        features = model(inputs)

        all_features.append(features.cpu().numpy())
        all_labels.append(labels)
        all_path.extend(path)

        if batch_id % 200 == 0:
            print("%d/%d" % (batch_id, len(dataloader) - 1))

all_features_cat = np.concatenate(all_features, 0)
all_labels_cat = np.concatenate(all_labels, 0)
all_paths_cat = np.stack(all_path)

np.save("./data/esogu_train_features.npy", all_features_cat)
np.save("./data/esogu_train_labels.npy", all_labels_cat)
np.save("./data/esogu_train_paths.npy", all_paths_cat)

all_features_cat = np.load("./data/esogu_train_features.npy")
all_labels_cat = np.load("./data/esogu_train_labels.npy")
all_paths_cat = np.load("./data/esogu_train_paths.npy")

n_subcenter = 7
feats_dim = 512
data = np.zeros(
    len(all_paths_cat),
    dtype={
        "names": ("path", "class", "subset", "subcluster"),
        "formats": ("U99", "i4", "i4", "i4"),
    },
)
data_sc = np.zeros(
    len(all_paths_cat),
    dtype={
        "names": ("path", "class", "subset", "subcluster"),
        "formats": ("U99", "i4", "i4", "i4"),
    },
)
data_kmedoids = np.zeros(
    len(all_paths_cat),
    dtype={
        "names": ("path", "class", "subset", "subcluster"),
        "formats": ("U99", "i4", "i4", "i4"),
    },
)
data_ward = np.zeros(
    len(all_paths_cat),
    dtype={
        "names": ("path", "class", "subset", "subcluster"),
        "formats": ("U99", "i4", "i4", "i4"),
    },
)

centers = np.zeros(
    (
        len(np.unique(all_labels_cat[:, 0])),
        n_subcenter * len(np.unique(all_labels_cat[:, 1])),
        feats_dim,
    )
)
centers_sc = np.zeros(
    (
        len(np.unique(all_labels_cat[:, 0])),
        n_subcenter * len(np.unique(all_labels_cat[:, 1])),
        feats_dim,
    )
)
centers_kmedoids = np.zeros(
    (
        len(np.unique(all_labels_cat[:, 0])),
        n_subcenter * len(np.unique(all_labels_cat[:, 1])),
        feats_dim,
    )
)
centers_ward = np.zeros(
    (
        len(np.unique(all_labels_cat[:, 0])),
        n_subcenter * len(np.unique(all_labels_cat[:, 1])),
        feats_dim,
    )
)

for indeks, selected_class in enumerate(np.unique(all_labels_cat[:, 0])):
    idxs_ = all_labels_cat[:, 0] == selected_class
    for selected_subset in np.unique(all_labels_cat[idxs_, 1]):
        idxs = (all_labels_cat[:, 0] == selected_class) * (
            all_labels_cat[:, 1] == selected_subset
        )

        normalized_feats = normalize(all_features_cat[idxs])

        kmeans_object = KMeans(n_clusters=n_subcenter, random_state=0).fit(
            normalized_feats
        )
        sc_object = SpectralClustering(
            n_clusters=n_subcenter, assign_labels="discretize", random_state=0
        ).fit(normalized_feats)
        kmedoids_object = KMedoids(n_clusters=n_subcenter).fit(normalized_feats)
        ward_object = AgglomerativeClustering(n_clusters=n_subcenter).fit(
            normalized_feats
        )

        subcluster_labels = kmeans_object.labels_
        subcluster_labels_sc = sc_object.labels_
        subcluster_labels_kmedoids = kmedoids_object.labels_
        subcluster_labels_ward = ward_object.labels_

        subcluster_centers_ = kmeans_object.cluster_centers_

        subcluster_centers = np.zeros_like(subcluster_centers_)
        subcluster_centers_sc = np.zeros_like(subcluster_centers_)
        subcluster_centers_kmedoids = np.zeros_like(subcluster_centers_)
        subcluster_centers_ward = np.zeros_like(subcluster_centers_)

        for i in range(n_subcenter):
            subcluster_centers[i, :] = all_features_cat[idxs][
                subcluster_labels == i
            ].mean(0)
            subcluster_centers_sc[i, :] = all_features_cat[idxs][
                subcluster_labels_sc == i
            ].mean(0)
            subcluster_centers_kmedoids[i, :] = all_features_cat[idxs][
                subcluster_labels_kmedoids == i
            ].mean(0)
            subcluster_centers_ward[i, :] = all_features_cat[idxs][
                subcluster_labels_ward == i
            ].mean(0)

        centers[selected_class][
            selected_subset * n_subcenter : selected_subset * n_subcenter + n_subcenter
        ] = subcluster_centers
        centers_sc[selected_class][
            selected_subset * n_subcenter : selected_subset * n_subcenter + n_subcenter
        ] = subcluster_centers_sc
        centers_kmedoids[selected_class][
            selected_subset * n_subcenter : selected_subset * n_subcenter + n_subcenter
        ] = subcluster_centers_kmedoids
        centers_ward[selected_class][
            selected_subset * n_subcenter : selected_subset * n_subcenter + n_subcenter
        ] = subcluster_centers_ward

        data["class"][idxs] = all_labels_cat[idxs, 0]
        data["subcluster"][idxs] = subcluster_labels
        data["path"][idxs] = all_paths_cat[idxs]
        data["subset"][idxs] = selected_subset

        data_sc["class"][idxs] = all_labels_cat[idxs, 0]
        data_sc["subcluster"][idxs] = subcluster_labels_sc
        data_sc["path"][idxs] = all_paths_cat[idxs]
        data_sc["subset"][idxs] = selected_subset

        data_kmedoids["class"][idxs] = all_labels_cat[idxs, 0]
        data_kmedoids["subcluster"][idxs] = subcluster_labels_kmedoids
        data_kmedoids["path"][idxs] = all_paths_cat[idxs]
        data_kmedoids["subset"][idxs] = selected_subset

        data_ward["class"][idxs] = all_labels_cat[idxs, 0]
        data_ward["subcluster"][idxs] = subcluster_labels_ward
        data_ward["path"][idxs] = all_paths_cat[idxs]
        data_ward["subset"][idxs] = selected_subset

        del normalized_feats

    if indeks % 100 == 0:
        print("%d" % (indeks))

np.save("./data/esogu_faces/train_meta_%d_clustured_kmeans.npy" % (n_subcenter), data)
np.save(
    "./data/esogu_faces/train_meta_%d_clustured_centers_kmeans.npy" % (n_subcenter),
    centers,
)

np.save("./data/esogu_faces/train_meta_%d_clustured_sc.npy" % (n_subcenter), data_sc)
np.save(
    "./data/esogu_faces/train_meta_%d_clustured_centers_sc.npy" % (n_subcenter),
    centers_sc,
)

np.save(
    "./data/esogu_faces/train_meta_%d_clustured_kmedoids.npy" % (n_subcenter),
    data_kmedoids,
)
np.save(
    "./data/esogu_faces/train_meta_%d_clustured_centers_kmedoids.npy" % (n_subcenter),
    centers_kmedoids,
)

np.save(
    "./data/esogu_faces/train_meta_%d_clustured_ward.npy" % (n_subcenter), data_ward
)
np.save(
    "./data/esogu_faces/train_meta_%d_clustured_centers_ward.npy" % (n_subcenter),
    centers_ward,
)
