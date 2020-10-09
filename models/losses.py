import torch
import torch.nn as nn
import math
from utils.iou import rotate_bbox_iou

class FocalLoss(nn.Module):

    def forward(self, cls, reg, anchors, annotations):
        alpha = 0.25
        gamma = 2.0
        batch_size = cls.shape[0]
        classification_losses = []
        regression_losses = []

        anchor = anchors[0, :, :]
        anchor_widths = anchor[:, 2]
        anchor_heights = anchor[:, 3]
        anchor_ctr_x = anchor[:, 0]
        anchor_ctr_y = anchor[:, 1]
        anchor_theta = anchor[:, 4]

        for j in range(batch_size):
            classification = cls[j, :, :]
            regression = reg[j, :, :]

            #bbox_annotation = annotations[j, :, :]
            bbox_annotation = annotations[j]
            #bbox_annotation = bbox_annotation[bbox_annotation[:, -1] != 0]

            classification = torch.clamp(classification, 1e-4, 1.0 - 1e-4)

            if bbox_annotation.shape[0] == 0:
                if torch.cuda.is_available():
                    alpha_factor = torch.ones(classification.shape).cuda() * alpha
                    alpha_factor = 1.0 - alpha_factor
                    focal_weight = classification
                    focal_weight = alpha_factor * torch.pow(focal_weight, gamma)

                    bce = -torch.log(1.0 - classification)

                    cls_loss = focal_weight * bce
                    classification_losses.append(cls_loss.sum())
                    regression_losses.append(torch.tensor(0).float())
                else:
                    alpha_factor = torch.ones(classification.shape) * alpha
                    alpha_factor = 1.0 - alpha_factor
                    focal_weight = classification
                    focal_weight = alpha_factor * torch.pow(focal_weight, gamma)

                    bce = -torch.log(1.0 - classification)

                    cls_loss = focal_weight * bce
                    classification_losses.append(cls_loss.sum())
                    regression_losses.append(torch.tensor(0).float())
                continue

            IoU = rotate_bbox_iou(anchor, bbox_annotation[:, :5])

            IoU_max, IoU_argmax = torch.max(IoU, dim=1)

            targets = torch.ones(classification.shape) * -1
            if torch.cuda.is_available():
                targets = targets.cuda()

            targets[torch.lt(IoU_max, 0.4), :] = 0
            positive_indices = torch.ge(IoU_max, 0.5)

            num_positive_anchors = positive_indices.sum()

            assigned_annotations = bbox_annotation[IoU_argmax, :]

            targets[positive_indices, :] = 0
            targets[positive_indices, assigned_annotations[positive_indices, 5].long()] = 1
            if torch.cuda.is_available():
                alpha_factor = torch.ones(targets.shape).cuda() * alpha
            else:
                alpha_factor = torch.ones(targets.shape) * alpha

            alpha_factor = torch.where(torch.eq(targets, 1.0), alpha_factor, 1.0 - alpha_factor)
            focal_weight = torch.where(torch.eq(targets, 1.0), 1.0 - classification, classification)
            focal_weight = alpha_factor * torch.pow(focal_weight, gamma)

            bce = -(targets * torch.log(classification)+ (1.0 - targets) * torch.log(1.0 - classification))

            cls_loss = focal_weight * bce
            if torch.cuda.is_available():
                cls_loss = torch.where(torch.ne(targets, -1.0), cls_loss, torch.zeros(cls_loss.shape).cuda())
            else:
                cls_loss = torch.where(torch.ne(targets, -1.0), cls_loss, torch.zeros(cls_loss.shape))

            classification_losses.append(cls_loss.sum() / torch.clamp(num_positive_anchors.float(), min=1.0))

            # regression loss
            if num_positive_anchors > 0:
                assigned_annotations = assigned_annotations[positive_indices, :]

                anchor_widths_pi = anchor_widths[positive_indices]
                anchor_heights_pi = anchor_heights[positive_indices]
                anchor_ctr_x_pi = anchor_ctr_x[positive_indices]
                anchor_ctr_y_pi = anchor_ctr_y[positive_indices]
                anchor_theta_pi = anchor_theta[positive_indices]

                gt_widths = assigned_annotations[:, 2]
                gt_heights = assigned_annotations[:, 3]
                gt_ctr_x = assigned_annotations[:, 0]
                gt_ctr_y = assigned_annotations[:, 1]
                gt_theta = assigned_annotations[:, 4]

                gt_widths = torch.clamp(gt_widths, min=1)
                gt_heights = torch.clamp(gt_heights, min=1)

                targets_dx = (gt_ctr_x - anchor_ctr_x_pi) / anchor_widths_pi
                targets_dy = (gt_ctr_y - anchor_ctr_y_pi) / anchor_heights_pi
                targets_dw = torch.log(gt_widths / anchor_widths_pi)
                targets_dh = torch.log(gt_heights / anchor_heights_pi)
                targets_theta = (gt_theta - anchor_theta_pi) * math.pi / 180.0

                targets = torch.stack((targets_dx, targets_dy, targets_dw, targets_dh, targets_theta))
                targets = targets.t()

                if torch.cuda.is_available():
                    targets = targets / torch.Tensor([[0.1, 0.1, 0.2, 0.2, 0.1]]).cuda()
                else:
                    targets = targets / torch.Tensor([[0.1, 0.1, 0.2, 0.2, 0.1]])

                regression_diff = torch.abs(targets - regression[positive_indices, :])

                reg_loss = torch.where(
                    torch.le(regression_diff, 1.0 * 9.0),
                    0.5 * 9.0 * torch.pow(regression_diff, 2),
                    regression_diff - 0.5 / 9.0
                )
                regression_losses.append(reg_loss.mean())

            else:
                if torch.cuda.is_available():
                    regression_losses.append(torch.tensor(0).float().cuda())
                else:
                    regression_losses.append(torch.tensor(0).float())

        return torch.stack(classification_losses).mean(dim=0, keepdim=True), torch.stack(regression_losses).mean(dim=0, keepdim=True)

