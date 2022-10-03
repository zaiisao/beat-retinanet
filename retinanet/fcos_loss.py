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

class FocalLoss(nn.Module):
    def forward(self, classifications, anchors, annotations):
        alpha = 0.25
        gamma = 2.0
        batch_size = classifications.shape[0]
        classification_losses = []

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

            #IoU = calc_iou(anchors[0, :, :], bbox_annotation[:, :2]) # num_anchors x num_annotations

            #IoU_max, IoU_argmax = torch.max(IoU, dim=1) # num_anchors x 1
            

            targets = torch.ones(classification.shape) * -1

            if torch.cuda.is_available():
                targets = targets.cuda()

            targets[torch.lt(IoU_max, 0.4), :] = 0

            positive_indices = torch.ge(IoU_max, 0.5)

            num_positive_anchors = positive_indices.sum()

            assigned_annotations = bbox_annotation[IoU_argmax, :]

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
    def forward(self, regressions, anchors, annotations, loss_type="f1"):
        batch_size = regressions.shape[0]
        regression_losses = []

        anchor = anchors[0, :, :]

        anchor_widths  = anchor[:, 1] - anchor[:, 0]
        anchor_ctr_x   = anchor[:, 0] + 0.5 * anchor_widths

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

            IoU = calc_iou(anchors[0, :, :], bbox_annotation[:, :2]) # num_anchors x num_annotations

            IoU_max, IoU_argmax = torch.max(IoU, dim=1) # num_anchors x 1

            positive_indices = torch.ge(IoU_max, 0.5)

            assigned_annotations = bbox_annotation[IoU_argmax, :]

            if positive_indices.sum() > 0:
                assigned_annotations = assigned_annotations[positive_indices, :]

                anchor_widths_pi = anchor_widths[positive_indices]
                #anchor_heights_pi = anchor_heights[positive_indices]
                anchor_ctr_x_pi = anchor_ctr_x[positive_indices]
                #anchor_ctr_y_pi = anchor_ctr_y[positive_indices]

                #gt_widths  = assigned_annotations[:, 2] - assigned_annotations[:, 0]
                gt_widths  = assigned_annotations[:, 1] - assigned_annotations[:, 0]
                #gt_heights = assigned_annotations[:, 3] - assigned_annotations[:, 1]
                gt_ctr_x   = assigned_annotations[:, 0] + 0.5 * gt_widths
                #gt_ctr_y   = assigned_annotations[:, 1] + 0.5 * gt_heights

                # clip widths to 1
                gt_widths  = torch.clamp(gt_widths, min=1)
                #gt_heights = torch.clamp(gt_heights, min=1)

                if loss_type == "f1":
                    targets_dx = (gt_ctr_x - anchor_ctr_x_pi) / anchor_widths_pi
                    #targets_dy = (gt_ctr_y - anchor_ctr_y_pi) / anchor_heights_pi
                    targets_dw = torch.log(gt_widths / anchor_widths_pi)
                    #targets_dh = torch.log(gt_heights / anchor_heights_pi)

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
                elif loss_type == "iou" or loss_type == "giou":
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

                    if loss_type == "iou":
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
