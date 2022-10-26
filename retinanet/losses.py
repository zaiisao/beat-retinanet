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

def get_fcos_positives(bbox_annotation, anchor, lower_limit, upper_limit):
    # sort from shortest to longest
    bbox_annotation = bbox_annotation[(bbox_annotation[:, 1] - bbox_annotation[:, 0]).argsort()]

    points_in_lines = torch.logical_and(
        torch.ge(torch.unsqueeze(anchor, dim=0), torch.unsqueeze(bbox_annotation[:, 0], dim=1)),
        torch.le(torch.unsqueeze(anchor, dim=0), torch.unsqueeze(bbox_annotation[:, 1], dim=1))
    )

    left = torch.unsqueeze(anchor, dim=0) - torch.unsqueeze(bbox_annotation[:, 0], dim=1)
    right = torch.unsqueeze(bbox_annotation[:, 1], dim=1) - torch.unsqueeze(anchor, dim=0)

    points_within_range = torch.logical_and(
        torch.max(left, right) < upper_limit,
        torch.max(left, right) >= lower_limit
    )

    positive_indices, positive_argmax = torch.logical_and(points_in_lines, points_within_range).max(dim=0)

    assigned_annotations = bbox_annotation[positive_argmax, :]

    left = torch.diagonal(left[positive_argmax], 0)
    right = torch.diagonal(right[positive_argmax], 0)

    return positive_indices, assigned_annotations, left, right

class FocalLoss(nn.Module):
    def __init__(self, fcos=False):
        super(FocalLoss, self).__init__()
        self.fcos = fcos

    # self.focalLoss(
    #     classification_outputs[feature_index],
    #     anchors[feature_index],
    #     annotations
    # )
    def forward(self, classifications, anchors, annotations, regress_limits=(0, float('inf'))):
        alpha = 0.25
        gamma = 2.0

        anchor = anchors[0, :, :]

        batch_size = classifications.shape[0]
        classification_losses = []

        if self.fcos:
            # anchors = (x, y) in feature map
            # [[x1, y1, x2, y2], [x1, y1, x2, y2], [x1, y1, x2, y2], []]
            assert torch.all(anchor[:, 0] == anchor[:, 1])
            anchor = anchor[:, 0]

        for j in range(batch_size):
            # j refers to an audio sample in batch
            jth_classification = classifications[j, :, :]

            # get box annotations from the original image
            # (5, 20, 0), (-1, -1, -1), 
            bbox_annotation = annotations[j, :, :]
            bbox_annotation = bbox_annotation[bbox_annotation[:, 2] != -1]

            jth_classification = torch.clamp(jth_classification, 1e-4, 1.0 - 1e-4)

            if bbox_annotation.shape[0] == 0:
                if torch.cuda.is_available():
                    alpha_factor = torch.ones(jth_classification.shape).cuda() * alpha

                    alpha_factor = 1. - alpha_factor
                    focal_weight = jth_classification
                    focal_weight = alpha_factor * torch.pow(focal_weight, gamma)

                    bce = -(torch.log(1.0 - jth_classification))

                    cls_loss = focal_weight * bce
                    classification_losses.append(cls_loss.sum())
                else:
                    alpha_factor = torch.ones(jth_classification.shape) * alpha

                    alpha_factor = 1. - alpha_factor
                    focal_weight = jth_classification
                    focal_weight = alpha_factor * torch.pow(focal_weight, gamma)

                    bce = -(torch.log(1.0 - jth_classification))

                    cls_loss = focal_weight * bce
                    classification_losses.append(cls_loss.sum())

                continue

            if self.fcos:
                targets = torch.zeros(jth_classification.shape)

                positive_indices, assigned_annotations, _, _ = get_fcos_positives(
                    bbox_annotation,
                    anchor,
                    regress_limits[0],
                    regress_limits[1]
                )
            else:
                targets = torch.ones(jth_classification.shape) * -1
                # print(f"targets shape in focal loss: {targets.shape}")
                # print(f"anchors shape in focal loss: {anchors.shape}")
                # print(f"anchors[0, :, :] shape in focal loss: {anchors[0, :, :].shape}")
                # print(f"bbox_annotation[:, :2] shape in focal loss: {bbox_annotation[:, :2].shape}")
                IoU = calc_iou(anchors[0, :, :], bbox_annotation[:, :2])
                # print(f"IoU shape in focal loss: {IoU.shape}")
                IoU_max, IoU_argmax = torch.max(IoU, dim=1) # num_anchors x 1
                # print(f"IoU_max shape in focal loss: {IoU_max.shape}")
                # print(f"IoU_argmax shape in focal loss: {IoU_argmax.shape}")

                targets[torch.lt(IoU_max, 0.4), :] = 0
                positive_indices = torch.ge(IoU_max, 0.5)

                assigned_annotations = bbox_annotation[IoU_argmax, :]

            if torch.cuda.is_available():
                targets = targets.cuda()

            num_positive_anchors = positive_indices.sum()

            targets[positive_indices, :] = 0
            targets[positive_indices, assigned_annotations[positive_indices, 2].long()] = 1

            if torch.cuda.is_available():
                alpha_factor = torch.ones(targets.shape).cuda() * alpha
            else:
                alpha_factor = torch.ones(targets.shape) * alpha

            alpha_factor = torch.where(torch.eq(targets, 1.), alpha_factor, 1. - alpha_factor)
            focal_weight = torch.where(torch.eq(targets, 1.), 1. - jth_classification, jth_classification)
            focal_weight = alpha_factor * torch.pow(focal_weight, gamma)

            bce = -(targets * torch.log(jth_classification) + (1.0 - targets) * torch.log(1.0 - jth_classification))

            cls_loss = focal_weight * bce   # in the case of the fifth feature map level,
                                            # shape of the cls_loss tensor is (W_5 * 3, 2) = (512 * 3, 2) = (1536, 2)
                                            # thus we have 1536 * 2 binary classifiers

            if torch.cuda.is_available():
                cls_loss = torch.where(torch.ne(targets, -1.0), cls_loss, torch.zeros(cls_loss.shape).cuda())
            else:
                cls_loss = torch.where(torch.ne(targets, -1.0), cls_loss, torch.zeros(cls_loss.shape))

            classification_losses.append(cls_loss.sum()/torch.clamp(num_positive_anchors.float(), min=1.0)) # the shape of cls_loss.sum() is []
        # END OF LOOP: for j in range(batch_size)
        # classification_losses is a list of numbers with the length of batch size

        if self.fcos:
            return torch.stack(classification_losses).sum(dim=0)
        else:
            # the shape of torch.stack(classification_losses) is (batch_size) and is a vector
            print(f"torch.stack(classification_losses) shape in focal loss forward: {torch.stack(classification_losses).shape}")
            print(f"torch.stack(classification_losses) in focal loss forward: {torch.stack(classification_losses)}")
            # torch.stack(classification_losses).mean(dim=0, keepdim=True) in focal loss forward: tensor([0.0710], grad_fn=<MeanBackward1>)
            print(f"torch.stack(classification_losses).mean(dim=0, keepdim=True) in focal loss forward: {torch.stack(classification_losses).mean(dim=0, keepdim=True)}")
            return torch.stack(classification_losses).mean(dim=0, keepdim=True) # shape of the returned tensor is (1,)

class RegressionLoss(nn.Module):
    def __init__(self, fcos=False, loss_type="f1", weight=10, num_anchors=3):
        super(RegressionLoss, self).__init__()
        self.fcos = fcos
        self.loss_type = loss_type
        self.weight = weight
        self.num_anchors = num_anchors

    def forward(self, regressions, anchors, annotations, regress_limits=(0, float('inf')),test=False, epoch_num=-1, iter_num=-1, feature_index=-1):
        # regressions is (B, C, W, H), with C = 4*num_anchors = 4*9
        # in our case, regressions is (B, C, W), with C = 2*num_anchors = 2*1
        batch_size = regressions.shape[0] 
        regression_losses = []

        # anchors = [[ [5, 5, 10, 10], [5, 5, 15, 15], ... ]]
        # in our case, anchors = [[ [5, 5], [10, 10], [15, 15] ... ]]
        anchor = anchors[0, :, :]

        anchor_widths  = anchor[:, 1] - anchor[:, 0] # if fcos is true, anchor_widths = 0
        anchor_ctr_x   = anchor[:, 0] + 0.5 * anchor_widths # if fcos is true, anchor_ctr_x = anchor[:, 0]

        if self.fcos:
            assert torch.all(anchor[:, 0] == anchor[:, 1])
            anchor = anchor[:, 0] # [5, 10, 15, ...]

        for j in range(batch_size):
            jth_regression = regressions[j, :, :] # j'th audio in the current batch

            bbox_annotation = annotations[j, :, :]
            bbox_annotation = bbox_annotation[bbox_annotation[:, 2] != -1] # bbox_annotation[:, 2] is the classification label

            if bbox_annotation.shape[0] == 0:
                if torch.cuda.is_available():
                    regression_losses.append(torch.tensor(0).float().cuda())
                else:
                    regression_losses.append(torch.tensor(0).float())

                continue

            if self.fcos:
                positive_indices, assigned_annotations, left, right = get_fcos_positives(
                    bbox_annotation,
                    anchor,
                    regress_limits[0],
                    regress_limits[1]
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
                # print("gt", gt_widths)
                # print("anchor", anchor_widths_pi)

                # clip widths to 1
                gt_widths  = torch.clamp(gt_widths, min=1)

                if self.loss_type == "f1":
                    # equations 6, 7, 8, 9 from R-CNN paper
                    targets_dx = (gt_ctr_x - anchor_ctr_x_pi) / anchor_widths_pi
                    targets_dw = torch.log(gt_widths / anchor_widths_pi)

                    # targets_dx shape:
                    # torch.Size([371])
                    # targets_dw shape:
                    # torch.Size([371])
                    # targets.shape before t():
                    # torch.Size([2, 371])
                    # targets.shape after t():
                    # torch.Size([371, 2])
                    # targets shape:
                    # torch.Size([371, 2])
                    print(f"epoch: {epoch_num}, iter: {iter_num} feature_index: {feature_index}, {j}th targets_dx shape:\n{targets_dx.shape}")
                    print(f"epoch: {epoch_num}, iter: {iter_num} feature_index: {feature_index}, {j}th targets_dw shape:\n{targets_dw.shape}")
                    targets = torch.stack((targets_dx, targets_dw)) # the shape of targets is (2, num of positive_indices)
                    print(f"epoch: {epoch_num}, iter: {iter_num} feature_index: {feature_index}, {j}th targets.shape before t():\n{targets.shape}")
                    targets = targets.t()# the shape of targets is (num of positive_indices, 2)
                    print(f"epoch: {epoch_num}, iter: {iter_num} feature_index: {feature_index}, {j}th targets.shape after t():\n{targets.shape}")

                    if torch.cuda.is_available():
                        targets = targets/torch.Tensor([[0.1, 0.2]]).cuda() # the shape of torch.Tensor([[0.1, 0.2]] is (1, 2)
                    else:
                        targets = targets/torch.Tensor([[0.1, 0.2]])

                    #if test:
                        #print(f"targets: {targets}")
                        #print(f"pred: {jth_regression[positive_indices, :]}")

                    negative_indices = 1 + (~positive_indices)

                    # targets shape:
                    # torch.Size([371, 2])
                    # jth_regression[positive_indices, :] shape:
                    # torch.Size([371, 2])
                    # total number of anchors: 1536
                    # total number of positive anchor indices: 371

                    print(f"epoch: {epoch_num}, iter: {iter_num} feature_index: {feature_index}, {j}th targets shape:\n{targets.shape}")
                    print(f"epoch: {epoch_num}, iter: {iter_num} feature_index: {feature_index}, {j}th_regression[positive_indices, :] shape:\n{jth_regression[positive_indices, :].shape}")
                    regression_diff = torch.abs(targets - jth_regression[positive_indices, :]) # the shape of regression_diff is (num of positive_indices, 2)
                    # the shape of jth_regression is (number of all anchors on the feature map, 2)
                    # the shape of jth_regression[positive_indices, :] is (num of positive_indices, 2)

                    if test:
                        print(f"epoch: {epoch_num}, iter: {iter_num} feature_index: {feature_index}, {j}th regression diff: {regression_diff[:, 0].mean(), regression_diff[:, 1].mean()}")

                    # on fifth level, jth_regression_loss tensor shape is (512 * 3 = 1536, 2)
                    # on fifth level, anchors[0, :, :] tensor shape is (1536, 2) (must be the same as regression loss)
                    jth_regression_loss = torch.where(
                        torch.le(regression_diff, 1.0 / 9.0),
                        0.5 * 9.0 * torch.pow(regression_diff, 2),
                        regression_diff - 0.5 / 9.0 # 9 is the square of sigma hyperparameter
                    )
                    # (number of positive anchors, 2)
                    torch.set_printoptions(edgeitems=10000)
                    print(f"epoch: {epoch_num}, iter: {iter_num} feature_index: {feature_index} total number of positive anchor boxes: {positive_indices.sum()}, PREDICTION: {j}th in regressionLoss forward:\n {jth_regression[positive_indices, :]}")
                    print(f"epoch: {epoch_num}, iter: {iter_num} feature_index: {feature_index} total number of positive anchor boxes: {positive_indices.sum()}, TARGETS: {j}th in regressionLoss forward:\n {targets}")
                    print(f"epoch: {epoch_num}, iter: {iter_num} feature_index: {feature_index} total number of positive anchor boxes: {positive_indices.sum()}, LOSS: {j}th in regressionLoss forward\n {jth_regression_loss}")
                    torch.set_printoptions(edgeitems=3)

                    regression_losses.append(jth_regression_loss.mean())
                elif self.loss_type == "iou" or self.loss_type == "giou":
                    #num_positive_anchors = positive_indices.sum()

                    # 1. For the predicted line B_p, ensuring  x_p_2 > x_p_1
                    # bbox_prediction, _ = torch.sort(jth_regression[positive_indices, :])
                    # bbox_ground = torch.stack((left, right), dim=1)[positive_indices, :].float()#assigned_annotations[:, :2]

                    # # 2. Calculating length of B_g: L_g = x_g_2 − x_g_1 (위에서 이미 정의한 gt_widths)
                    # bbox_ground_lengths = bbox_ground[:, 1] - bbox_ground[:, 0] #gt_widths

                    # # 3. Calculating length of B_p: L_p = x̂_p_2 − x̂_p_1
                    # bbox_prediction_lengths = bbox_prediction[:, 1] - bbox_prediction[:, 0]

                    # # 4. Calculating intersection I between B_p and B_g
                    # intersection_x1 = torch.max(bbox_ground[:, 0], bbox_prediction[:, 0])
                    # intersection_x2 = torch.min(bbox_ground[:, 1], bbox_prediction[:, 1])
                    # intersection = torch.where(
                    #     intersection_x2 > intersection_x1,
                    #     intersection_x2 - intersection_x1,
                    #     torch.zeros(bbox_ground.size(dim=0))
                    # )

                    # # 5. Finding the coordinate of smallest enclosing line B_c:
                    # coordinate_x1 = torch.min(bbox_ground[:, 0], bbox_prediction[:, 0])
                    # coordinate_x2 = torch.max(bbox_ground[:, 1], bbox_prediction[:, 1])

                    # # 6. Calculating length of B_c
                    # bbox_coordinate = coordinate_x2 - coordinate_x1 + 1e-7

                    # # 7. IoU (I / U), where U = L_p + L_g - I
                    # union = bbox_prediction_lengths + bbox_ground_lengths - intersection
                    # iou = intersection / union

                    # if self.loss_type == "iou":
                    #     # 9a. L_IoU = 1 - IoU
                    #     regression_loss = 1 - iou
                    # else:
                    #     # 8. GIoU = IoU - (L_c - U)/L_c
                    #     giou = iou - (bbox_coordinate - union)/bbox_coordinate
                    #     print(bbox_prediction, bbox_ground, giou)

                    #     # 9b. L_GIoU = 1 - GIoU
                    #     regression_loss = 1 - giou

                    # #print(regression_loss.mean(), torch.exp(regression_loss.mean() * self.weight))
                    # regression_losses.append(regression_loss.mean() * self.weight)
                    target_left = left[positive_indices]
                    target_right = right[positive_indices]

                    pred_left = jth_regression[positive_indices, 0]
                    pred_right = jth_regression[positive_indices, 1]

                    target_area = (target_left + target_right)
                    pred_area = (pred_left + pred_right)

                    w_intersect = torch.min(pred_left, target_left) + torch.min(pred_right, target_right)
                    g_w_intersect = torch.max(pred_left, target_left) + torch.max(pred_right, target_right)
                    ac_uion = g_w_intersect + 1e-7
                    area_intersect = w_intersect
                    area_union = target_area + pred_area - area_intersect
                    ious = (area_intersect + 1.0) / (area_union + 1.0)
                    gious = ious - (ac_uion - area_union) / ac_uion
                    if self.loss_type == 'iou':
                        losses = -torch.log(ious)
                    elif self.loss_type == 'linear_iou':
                        losses = 1 - ious
                    elif self.loss_type == 'giou':
                        losses = 1 - gious
                    else:
                        raise NotImplementedError
                    #print(torch.stack((target_left, target_right), dim=1), torch.stack((pred_left, pred_right), dim=1), losses)

                    # if weight is not None and weight.sum() > 0:
                    #     return (losses * weight).sum()
                    # else:
                    #     assert losses.numel() != 0
                    #     return losses.sum()

                    regression_losses.append(losses.sum() * self.weight)

            else:
                if torch.cuda.is_available():
                    regression_losses.append(torch.tensor(0).float().cuda())
                else:
                    regression_losses.append(torch.tensor(0).float())

        print(f"epoch: {epoch_num}, iter: {iter_num} feature_index: {feature_index} torch.stack(regression_losses): {torch.stack(regression_losses)}")
        return torch.stack(regression_losses).mean(dim=0, keepdim=True) # return the regression loss averaged over the batch

class CenternessLoss(nn.Module):
    def __init__(self, fcos=False):
        super(CenternessLoss, self).__init__()
        self.fcos = fcos

    def forward(self, centernesses, anchors, annotations, regress_limits=(0, float('inf'))):
        if not self.fcos:
            raise NotImplementedError

        batch_size = centernesses.shape[0]
        centerness_losses = []

        anchor = anchors[0, :, :]

        assert torch.all(anchor[:, 0] == anchor[:, 1])
        anchor = anchor[:, 0]

        for j in range(batch_size):
            jth_centerness = centernesses[j, :, :]

            bbox_annotation = annotations[j, :, :]
            bbox_annotation = bbox_annotation[bbox_annotation[:, 2] != -1]

            #jth_centerness = torch.clamp(jth_centerness, 1e-4, 1.0 - 1e-4)
            jth_centerness = torch.sigmoid(jth_centerness)

            positive_indices, assigned_annotations, left, right = get_fcos_positives(
                bbox_annotation,
                anchor,
                regress_limits[0],
                regress_limits[1]
            )

            num_positive_anchors = positive_indices.sum()

            targets = torch.where(
                positive_indices,
                torch.sqrt(torch.min(left, right) / torch.max(left, right)).float(),
                torch.zeros(positive_indices.shape)
            ).unsqueeze(dim=1)

            bce = -(targets * torch.log(jth_centerness) + (1.0 - targets) * torch.log(1.0 - jth_centerness))

            if torch.cuda.is_available():
                ctr_loss = torch.where(positive_indices, bce, torch.zeros(bce.shape).cuda())
            else:
                ctr_loss = torch.where(positive_indices, bce, torch.zeros(bce.shape))

            #print(ctr_loss.sum()/torch.clamp(num_positive_anchors.float(), min=1.0))
            centerness_losses.append(ctr_loss.sum()/torch.clamp(num_positive_anchors.float(), min=1.0))

        if self.fcos:
            return torch.stack(centerness_losses).sum(dim=0)
        else:
            return torch.stack(centerness_losses).mean(dim=0, keepdim=True)
