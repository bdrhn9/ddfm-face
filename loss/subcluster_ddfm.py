import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class subcluster_ddfm_loss(nn.Module):
    """Center loss with subcenters added.

    Args:
        num_classes (int): number of classes.
        num_subcenters (int): number of subcenters per class
        feat_dim (int): feature dimension.
    """

    def __init__(
        self,
        num_classes=285,
        num_subset=1,
        num_subcluster=3,
        feat_dim=2,
        precalc_centers=None,
        margin=1.0,
    ):
        super(subcluster_ddfm_loss, self).__init__()

        self.num_classes = num_classes
        self.num_subset = num_subset
        self.num_subcenters = num_subset * num_subcluster
        self.num_subcluster = num_subcluster
        self.num_centers = self.num_classes * self.num_subcenters

        self.feat_dim = feat_dim
        self.margin = margin

        if precalc_centers:
            precalculated_centers = np.load(precalc_centers)
        with torch.no_grad():
            self.centers = nn.Parameter(
                torch.randn(
                    self.num_classes,
                    self.num_subcenters,
                    self.feat_dim,
                    requires_grad=True,
                )
            )
            if precalc_centers:
                self.centers.copy_(torch.from_numpy(precalculated_centers))
                print("Centers loaded.")

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)

        centers_batch = self.centers.view(-1, self.feat_dim).index_select(
            0, labels.long()
        )

        intraclass_dist = (
            torch.pow(x, 2).sum(dim=1, keepdim=True).expand(x.size(0), x.size(0))
            + torch.pow(centers_batch, 2)
            .sum(dim=1, keepdim=True)
            .expand(x.size(0), x.size(0))
            .t()
        )
        intraclass_dist.addmm_(x, centers_batch.t(), beta=1, alpha=-2)
        intraclass_loss = torch.diag(intraclass_dist).sum() / (
            batch_size * self.feat_dim * 2
        )
        intraclass_distances = torch.diag(intraclass_dist)

        centers_dist_inter = (
            torch.pow(x, 2).sum(dim=1, keepdim=True).expand(x.size(0), self.num_centers)
            + torch.pow(self.centers.view(-1, self.feat_dim), 2)
            .sum(dim=1, keepdim=True)
            .expand(self.num_centers, x.size(0))
            .t()
        )
        centers_dist_inter.addmm_(
            x, self.centers.view(-1, self.feat_dim).t(), beta=1, alpha=-2
        )
        centers_dist_inter = (
            intraclass_distances.repeat(self.num_centers, 1).t() - centers_dist_inter
        )

        uniq_cls_labels, uniq_cls_counts = (labels // self.num_subcenters).unique(
            return_counts=True
        )
        labels2care = uniq_cls_labels

        mask = torch.ones_like(
            centers_dist_inter, dtype=torch.bool, requires_grad=False
        )
        same_cls_labels = labels.long()[
            (labels // self.num_subcenters) == labels2care[0]
        ].unique()
        cls_mask = labels.eq(same_cls_labels[0])
        for same_cls_label in same_cls_labels[1:]:
            cls_mask += labels.eq(same_cls_label)
        for same_cls_label in same_cls_labels:
            mask[cls_mask, same_cls_label] = False
        # del mask

        for label in labels2care[1:]:
            # mask = torch.zeros_like(centers_dist_inter,dtype=torch.bool,requires_grad=False)
            same_cls_labels = labels.long()[
                (labels // self.num_subcenters) == label
            ].unique()
            cls_mask = labels.eq(same_cls_labels[0])
            for same_cls_label in same_cls_labels[1:]:
                cls_mask += labels.eq(same_cls_label)
            for same_cls_label in same_cls_labels:
                mask[cls_mask, same_cls_label] = False

        interclass_loss_triplet = (1 / (2.0 * self.num_centers * batch_size)) * (
            (self.margin + centers_dist_inter).clamp(min=0) * mask
        ).sum()

        unique_labels = labels.long().unique()
        unique_centers_batch = self.centers.view(-1, self.feat_dim).index_select(
            0, unique_labels
        )

        centers_dist = (
            torch.pow(unique_centers_batch, 2)
            .sum(dim=1, keepdim=True)
            .expand(unique_centers_batch.size(0), self.num_centers)
            + torch.pow(self.centers.view(-1, self.feat_dim), 2)
            .sum(dim=1, keepdim=True)
            .expand(self.num_centers, unique_centers_batch.size(0))
            .t()
        )
        centers_dist.addmm_(
            unique_centers_batch,
            self.centers.view(-1, self.feat_dim).t(),
            beta=1,
            alpha=-2,
        )

        uniq_cls_labels, uniq_cls_counts = (
            unique_labels // self.num_subcenters
        ).unique(return_counts=True)
        labels2care = uniq_cls_labels[uniq_cls_counts > 1]

        mask = torch.zeros_like(centers_dist, dtype=torch.bool, requires_grad=False)
        same_cls_labels = unique_labels[
            (unique_labels // self.num_subcenters) == labels2care[0]
        ].unique()
        cls_mask = unique_labels.eq(same_cls_labels[0])
        for same_cls_label in same_cls_labels[1:]:
            cls_mask += unique_labels.eq(same_cls_label)
        mask[
            cls_mask,
            same_cls_labels.expand(
                same_cls_labels.size(0), same_cls_labels.size(0)
            ).t(),
        ] = True

        interclass_loss = (1 / (self.num_centers * batch_size * 2.0)) * (
            (self.margin + centers_dist[mask].max() - centers_dist[cls_mask]).clamp(
                min=0
            )
            * torch.logical_not(mask)[cls_mask]
        ).sum()
        del mask

        for label in labels2care[1:]:
            mask = torch.zeros_like(centers_dist, dtype=torch.bool, requires_grad=False)
            same_cls_labels = unique_labels[
                (unique_labels // self.num_subcenters) == label
            ].unique()
            cls_mask = unique_labels.eq(same_cls_labels[0])
            for same_cls_label in same_cls_labels[1:]:
                cls_mask += unique_labels.eq(same_cls_label)
            mask[
                cls_mask,
                same_cls_labels.expand(
                    same_cls_labels.size(0), same_cls_labels.size(0)
                ).t(),
            ] = True
            interclass_loss += (1 / (self.num_centers * batch_size * 2.0)) * (
                (self.margin + centers_dist[mask].max() - centers_dist[cls_mask]).clamp(
                    min=0
                )
                * torch.logical_not(mask)[cls_mask]
            ).sum()
            del mask

        return intraclass_loss, interclass_loss, interclass_loss_triplet
