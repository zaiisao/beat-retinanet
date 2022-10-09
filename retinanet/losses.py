import numpy as np
import torch
import torch.nn as nn

def calc_iou(a, b):
    area = b[:, 1] - b[:, 0]

    iw = torch.min(torch.unsqueeze(a[:, 1], dim=1), b[:, 1]) - torch.max(torch.unsqueeze(a[:, 0], 1), b[:, 0])
    iw = torch.clamp(iw, min=0)

    ua = torch.unsqueeze(a[:, 1] - a[:, 0], dim=1) + area - iw
    ua = torch.clamp(ua, min=1e-8)

    intersection = iw

    IoU = intersection / ua

    return IoU

def get_fcos_positives(lines, points, lower_limit, upper_limit):
    # sort from shortest to longest
    lines = lines[(lines[:, 1] - lines[:, 0]).argsort()]

    points_in_lines = torch.logical_and(
        torch.ge(torch.unsqueeze(points, dim=0), torch.unsqueeze(lines[:, 0], dim=1)),
        torch.le(torch.unsqueeze(points, dim=0), torch.unsqueeze(lines[:, 1], dim=1))
    )

    left = torch.unsqueeze(points, dim=0) - torch.unsqueeze(lines[:, 0], dim=1)
    right = torch.unsqueeze(lines[:, 1], dim=1) - torch.unsqueeze(points, dim=0)
    
    points_within_range = torch.logical_and(
        torch.max(left, right) < upper_limit,
        torch.max(left, right) >= lower_limit
    )

    positive_indices, positive_argmax = torch.logical_and(points_in_lines, points_within_range).max(dim=0)

    assigned_annotations = lines[positive_argmax, :]

    left = torch.diagonal(left[positive_argmax], 0)
    right = torch.diagonal(right[positive_argmax], 0)

    return positive_indices, assigned_annotations, left, right

class FocalLoss(nn.Module):
    def __init__(self, fcos=False):
        super(FocalLoss, self).__init__()
        self.fcos = fcos

    def forward(self, classifications, anchors, annotations, limits=(0, float('inf')), centerness=None):
        alpha = 0.25
        gamma = 2.0
        batch_size = classifications.shape[0]
        classification_losses = []

        if self.fcos:
            assert torch.all(anchors[0, :, 0] == anchors[0, :, 1])
            anchors = anchors[0, :, 0]

        for j in range(batch_size):
            classification = classifications[j, :, :]

            bbox_annotation = annotations[j, :, :]
            bbox_annotation = bbox_annotation[bbox_annotation[:, 2] != -1]

            classification = torch.clamp(classification, 1e-4, 1.0 - 1e-4)

            if bbox_annotation.shape[0] == 0:
                if torch.cuda.is_available():
                    alpha_factor = torch.ones(classification.shape).cuda() * alpha

                    alpha_factor = 1. - alpha_factor
                    focal_weight = classification
                    focal_weight = alpha_factor * torch.pow(focal_weight, gamma)

                    bce = -(torch.log(1.0 - classification))

                    cls_loss = focal_weight * bce
                    classification_losses.append(cls_loss.sum())
                else:
                    alpha_factor = torch.ones(classification.shape) * alpha

                    alpha_factor = 1. - alpha_factor
                    focal_weight = classification
                    focal_weight = alpha_factor * torch.pow(focal_weight, gamma)

                    bce = -(torch.log(1.0 - classification))

                    cls_loss = focal_weight * bce
                    classification_losses.append(cls_loss.sum())

                continue

            targets = torch.ones(classification.shape) * -1

            if torch.cuda.is_available():
                targets = targets.cuda()

            if self.fcos:
                targets[:, :] = 0

                positive_indices, assigned_annotations, _, _ = get_fcos_positives(
                    bbox_annotation,
                    anchors,
                    limits[0],
                    limits[1]
                )
            else:
                IoU = calc_iou(anchors[0, :, :], bbox_annotation[:, :2])
                IoU_max, IoU_argmax = torch.max(IoU, dim=1) # num_anchors x 1

                targets[torch.lt(IoU_max, 0.4), :] = 0
                positive_indices = torch.ge(IoU_max, 0.5)

                assigned_annotations = bbox_annotation[IoU_argmax, :]

            num_positive_anchors = positive_indices.sum()

            targets[positive_indices, :] = 0
            targets[positive_indices, assigned_annotations[positive_indices, 2].long()] = 1

            if torch.cuda.is_available():
                alpha_factor = torch.ones(targets.shape).cuda() * alpha
            else:
                alpha_factor = torch.ones(targets.shape) * alpha

            alpha_factor = torch.where(torch.eq(targets, 1.), alpha_factor, 1. - alpha_factor)
            focal_weight = torch.where(torch.eq(targets, 1.), 1. - classification, classification)
            focal_weight = alpha_factor * torch.pow(focal_weight, gamma)

            bce = -(targets * torch.log(classification) + (1.0 - targets) * torch.log(1.0 - classification))

            cls_loss = focal_weight * bce

            if torch.cuda.is_available():
                cls_loss = torch.where(torch.ne(targets, -1.0), cls_loss, torch.zeros(cls_loss.shape).cuda())
            else:
                cls_loss = torch.where(torch.ne(targets, -1.0), cls_loss, torch.zeros(cls_loss.shape))

            classification_losses.append(cls_loss.sum()/torch.clamp(num_positive_anchors.float(), min=1.0))

        return torch.stack(classification_losses).mean(dim=0, keepdim=True)

class RegressionLoss(nn.Module):
    def __init__(self, fcos=False, loss_type="f1"):
        super(RegressionLoss, self).__init__()
        self.fcos = fcos
        self.loss_type = loss_type

    def forward(self, regressions, anchors, annotations, limits=(0, float('inf'))):
        batch_size = regressions.shape[0]
        regression_losses = []

        anchor = anchors[0, :, :]

        anchor_widths  = anchor[:, 1] - anchor[:, 0]
        anchor_ctr_x   = anchor[:, 0] + 0.5 * anchor_widths

        if self.fcos:
            assert torch.all(anchors[0, :, 0] == anchors[0, :, 1])
            anchors = anchors[0, :, 0]

        for j in range(batch_size):
            regression = regressions[j, :, :]

            bbox_annotation = annotations[j, :, :]
            bbox_annotation = bbox_annotation[bbox_annotation[:, 2] != -1]

            if bbox_annotation.shape[0] == 0:
                if torch.cuda.is_available():
                    regression_losses.append(torch.tensor(0).float().cuda())
                else:
                    regression_losses.append(torch.tensor(0).float())

                continue

            if self.fcos:
                positive_indices, assigned_annotations, _, _ = get_fcos_positives(
                    bbox_annotation,
                    anchors,
                    limits[0],
                    limits[1]
                )
            else:
                IoU = calc_iou(anchors[0, :, :], bbox_annotation[:, :2]) # num_anchors x num_annotations
                IoU_max, IoU_argmax = torch.max(IoU, dim=1) # num_anchors x 1
                positive_indices = torch.ge(IoU_max, 0.5)
                assigned_annotations = bbox_annotation[IoU_argmax, :]

            if positive_indices.sum() > 0:
                assigned_annotations = assigned_annotations[positive_indices, :]

                anchor_widths_pi = anchor_widths[positive_indices]
                anchor_ctr_x_pi = anchor_ctr_x[positive_indices]

                gt_widths  = assigned_annotations[:, 1] - assigned_annotations[:, 0]
                gt_ctr_x   = assigned_annotations[:, 0] + 0.5 * gt_widths

                # clip widths to 1
                gt_widths  = torch.clamp(gt_widths, min=1)

                if self.loss_type == "f1":
                    targets_dx = (gt_ctr_x - anchor_ctr_x_pi) / anchor_widths_pi
                    targets_dw = torch.log(gt_widths / anchor_widths_pi)

                    targets = torch.stack((targets_dx, targets_dw))
                    targets = targets.t()

                    if torch.cuda.is_available():
                        targets = targets/torch.Tensor([[0.1, 0.2]]).cuda()
                    else:
                        targets = targets/torch.Tensor([[0.1, 0.2]])

                    negative_indices = 1 + (~positive_indices)

                    regression_diff = torch.abs(targets - regression[positive_indices, :])

                    regression_loss = torch.where(
                        torch.le(regression_diff, 1.0 / 9.0),
                        0.5 * 9.0 * torch.pow(regression_diff, 2),
                        regression_diff - 0.5 / 9.0
                    )
                elif self.loss_type == "iou" or self.loss_type == "giou":
                    # 1. For the predicted line B_p, ensuring  x_p_2 > x_p_1
                    bl_prediction, _ = torch.sort(regression[positive_indices, :])
                    bl_ground = assigned_annotations[:, :2]

                    # 2. Calculating length of B_g: L_g = x_g_2 − x_g_1 (위에서 이미 정의한 gt_widths)
                    bl_ground_lengths = gt_widths

                    # 3. Calculating length of B_p: L_p = x̂_p_2 − x̂_p_1
                    bl_prediction_lengths = bl_prediction[:, 1] - bl_prediction[:, 0]

                    # 4. Calculating intersection I between B_p and B_g
                    intersection_x1 = torch.max(bl_ground[:, 0], bl_prediction[:, 0])
                    intersection_x2 = torch.min(bl_ground[:, 1], bl_prediction[:, 1])
                    intersection = torch.where(
                        intersection_x2 > intersection_x1,
                        intersection_x2 - intersection_x1,
                        torch.zeros(bl_ground.size(dim=0))
                    )

                    # 5. Finding the coordinate of smallest enclosing line B_c:
                    coordinate_x1 = torch.min(bl_ground[:, 0], bl_prediction[:, 0])
                    coordinate_x2 = torch.max(bl_ground[:, 1], bl_prediction[:, 1])

                    # 6. Calculating length of B_c
                    bl_coordinate = coordinate_x2 - coordinate_x1

                    # 7. IoU (I / U), where U = L_p + L_g - I
                    union = bl_prediction_lengths + bl_ground_lengths - intersection
                    iou = intersection / union

                    if self.loss_type == "iou":
                        # 9a. L_IoU = 1 - IoU
                        regression_loss = 1 - iou
                    else:
                        # 8. GIoU = IoU - (L_c - U)/L_c
                        giou = iou - (bl_coordinate - union)/bl_coordinate

                        # 9b. L_GIoU = 1 - GIoU
                        regression_loss = 1 - giou

                regression_losses.append(regression_loss.mean())
            else:
                if torch.cuda.is_available():
                    regression_losses.append(torch.tensor(0).float().cuda())
                else:
                    regression_losses.append(torch.tensor(0).float())

        return torch.stack(regression_losses).mean(dim=0, keepdim=True)

class CenternessLoss(nn.Module):
    def __init__(self, fcos=False):
        super(CenternessLoss, self).__init__()
        self.fcos = fcos

    def forward(self, centernesses, anchors, annotations, limits=(0, float('inf'))):
        if not self.fcos:
            raise NotImplementedError

        batch_size = centernesses.shape[0]
        centerness_losses = []

        assert torch.all(anchors[0, :, 0] == anchors[0, :, 1])
        anchors = anchors[0, :, 0]

        for j in range(batch_size):
            centerness = centernesses[j, :, :]

            bbox_annotation = annotations[j, :, :]
            bbox_annotation = bbox_annotation[bbox_annotation[:, 2] != -1]

            centerness = torch.clamp(centerness, 1e-4, 1.0 - 1e-4)

            #targets = torch.ones(centerness.shape) * -1

            positive_indices, assigned_annotations, left, right = get_fcos_positives(
                bbox_annotation,
                anchors,
                limits[0],
                limits[1]
            )

            num_positive_anchors = positive_indices.sum()

            targets = torch.where(
                positive_indices,
                torch.sqrt(torch.min(left, right) / torch.max(left, right)).float(),
                torch.ones(positive_indices.shape) * -1
            ).unsqueeze(dim=1)

            bce = -(targets * torch.log(centerness) + (1.0 - targets) * torch.log(1.0 - centerness))

            if torch.cuda.is_available():
                ctr_loss = torch.where(torch.ne(targets, -1.0), bce, torch.zeros(bce.shape).cuda())
            else:
                ctr_loss = torch.where(torch.ne(targets, -1.0), bce, torch.zeros(bce.shape))

            centerness_losses.append(ctr_loss.sum()/torch.clamp(num_positive_anchors.float(), min=1.0))

        return torch.stack(centerness_losses).mean(dim=0, keepdim=True)
