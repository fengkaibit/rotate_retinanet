import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import math
from models.anchors import Anchors
from models.losses import FocalLoss
from utils.box_utils import BBoxTransform, ClipBoxes
from utils.nms import rotate_nms

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

class BottleNeck(nn.Module):
    expansion = 4

    def __init__(self, input, output, stride=1, downsample=None):
        super(BottleNeck, self).__init__()
        self.conv1 = nn.Conv2d(input, output, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(output)
        self.conv2 = nn.Conv2d(output, output, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(output)
        self.conv3 = nn.Conv2d(output, output * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(output * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(residual)

        out += residual
        out = self.relu(out)

        return out

class FPN(nn.Module):

    def __init__(self, C3_size, C4_size, C5_size, feature_size=256):
        super(FPN, self).__init__()

        self.p3_1 = nn.Conv2d(C3_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.p3_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        self.p4_1 = nn.Conv2d(C4_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.p4_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.p4_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        self.p5_1 = nn.Conv2d(C5_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.p5_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.p5_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        self.p6 = nn.Conv2d(C5_size, feature_size, kernel_size=3, stride=2, padding=1)

        self.p7_1 = nn.ReLU()
        self.p7_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=2, padding=1)

    def forward(self, inputs):
        C3, C4, C5 = inputs

        p5_out = self.p5_1(C5)
        p5_upsample_out = self.p5_upsample(p5_out)
        p5_out = self.p5_2(p5_out)

        p4_out = self.p4_1(C4)
        p4_out = p5_upsample_out + p4_out
        p4_upsample_out = self.p4_upsample(p4_out)
        p4_out = self.p4_2(p4_out)

        p3_out = self.p3_1(C3)
        p3_out = p4_upsample_out + p3_out
        p3_out = self.p3_2(p3_out)

        p6_out = self.p6(C5)

        p7_out = self.p7_1(p6_out)
        p7_out = self.p7_2(p7_out)

        return [p3_out, p4_out, p5_out, p6_out, p7_out]

class ClassficationModel(nn.Module):

    def __init__(self, num_features_in, num_anchors=27, num_classes=2, feature_size=256):
        super(ClassficationModel, self).__init__()

        self.num_anchors = num_anchors
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3,stride=1, padding=1)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)
        self.act4 = nn.ReLU()

        self.output = nn.Conv2d(feature_size, num_anchors * num_classes, kernel_size=3, padding=1)
        self.output_act = nn.Sigmoid()

    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.act2(out)

        out = self.conv3(out)
        out = self.act3(out)

        out = self.conv4(out)
        out = self.act4(out)

        out = self.output(out)
        out = self.output_act(out)

        # out: B, C, H, W
        out1 = out.permute(0, 2, 3, 1)
        B, H, W, C = out1.shape
        out2 = out1.view(B, H, W, self.num_anchors, self.num_classes)

        return out2.contiguous().view(x.shape[0], -1, self.num_classes)

class RegressionModel(nn.Module):
    def __init__(self, num_features_in, num_anchors=27, feature_size=256):
        super(RegressionModel, self).__init__()

        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, stride=1, padding=1)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)
        self.act4 = nn.ReLU()

        self.output = nn.Conv2d(feature_size, num_anchors * 5, kernel_size=3, padding=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.act2(out)

        out = self.conv3(out)
        out = self.act3(out)

        out = self.conv4(out)
        out = self.act4(out)

        out = self.output(out)

        out1 = out.permute(0, 2, 3, 1)

        return out1.contiguous().view(x.shape[0], -1, 5)

class RetinaNet(nn.Module):

    def __init__(self, num_classes, block, layer_nums):
        # 2, bottleNick, [3, 4, 6, 3]
        super(RetinaNet, self).__init__()
        self.input = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layer_nums[0])
        self.layer2 = self._make_layer(block, 128, layer_nums[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layer_nums[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layer_nums[3], stride=2)

        fpn_size = [self.layer2[-1].conv3.out_channels, self.layer3[-1].conv3.out_channels,
                    self.layer4[-1].conv3.out_channels]

        self.fpn = FPN(fpn_size[0], fpn_size[1], fpn_size[2])
        self.cls_model = ClassficationModel(256, num_classes=num_classes)
        self.reg_model = RegressionModel(256)

        for module in [self.cls_model, self.reg_model]:
            for layer in module.modules():
                if isinstance(layer, nn.Conv2d):
                    n = layer.kernel_size[0] * layer.kernel_size[1] * layer.out_channels
                    torch.nn.init.normal_(layer.weight, mean=0, std=math.sqrt(2. / n))
                    torch.nn.init.constant_(layer.bias, 0)

        prior = 0.01
        bias_value = -math.log((1 - prior) / prior)
        torch.nn.init.constant_(self.cls_model.output.bias, bias_value)

        self.freeze_bn()

        self.anchors = Anchors()
        self.focalLoss = FocalLoss()

        self.regressBoxes = BBoxTransform()
        self.clipBoxes = ClipBoxes()

    def _make_layer(self, block, output, layer_nums, stride=1):
        downsample = None
        if stride != 1 or self.input != output * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.input, output * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(output * block.expansion)
            )

        layers = [block(self.input, output, stride, downsample)]
        self.input = output * block.expansion
        for i in range(1, layer_nums):
            layers.append(block(self.input, output))

        return nn.Sequential(*layers)

    def freeze_bn(self):
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()

    def forward(self, inputs):
        if self.training:
            imgs, annotations = inputs
        else:
            imgs = inputs

        x = self.conv1(imgs)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        features = self.fpn([x2, x3, x4])

        cls = torch.cat([self.cls_model(feature) for feature in features], dim=1)
        reg = torch.cat([self.reg_model(feature) for feature in features], dim=1)

        anchors = self.anchors(imgs)

        if self.training:
            return self.focalLoss(cls, reg, anchors, annotations)
        else:
            transformed_anchors = self.regressBoxes(anchors, reg)
            transformed_anchors = self.clipBoxes(transformed_anchors, imgs)

            finalResult = [[], [], []]

            finalScores = torch.Tensor([])
            finalAnchorBoxesIndexes = torch.Tensor([]).long()
            finalAnchorBoxesCoordinates = torch.Tensor([])

            if torch.cuda.is_available():
                finalScores = finalScores.cuda()
                finalAnchorBoxesIndexes = finalAnchorBoxesIndexes.cuda()
                finalAnchorBoxesCoordinates = finalAnchorBoxesCoordinates.cuda()

            for i in range(cls.shape[2]):
                scores = torch.squeeze(cls[:, :, i])
                scores_over_thresh = (scores > 0.05)
                if scores_over_thresh.sum() == 0:
                    continue

                scores = scores[scores_over_thresh]
                anchorBoxes = torch.squeeze(transformed_anchors)
                anchorBoxes = anchorBoxes[scores_over_thresh]
                anchors_nms_idx = rotate_nms(anchorBoxes, scores, 0.5)

                finalResult[0].extend(scores[anchors_nms_idx])
                finalResult[1].extend(torch.tensor([i] * anchors_nms_idx.shape[0]))
                finalResult[2].extend(anchorBoxes[anchors_nms_idx])

                finalScores = torch.cat((finalScores, scores[anchors_nms_idx]))
                finalAnchorBoxesIndexesValue = torch.tensor([i] * anchors_nms_idx.shape[0])
                if torch.cuda.is_available():
                    finalAnchorBoxesIndexesValue = finalAnchorBoxesIndexesValue.cuda()

                finalAnchorBoxesIndexes = torch.cat((finalAnchorBoxesIndexes, finalAnchorBoxesIndexesValue))
                finalAnchorBoxesCoordinates = torch.cat((finalAnchorBoxesCoordinates, anchorBoxes[anchors_nms_idx]))

            return [finalScores, finalAnchorBoxesIndexes, finalAnchorBoxesCoordinates]


def build_retinanet(num_classes=2, pretrained=False, **kwargs):
    model = RetinaNet(num_classes, BottleNeck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']), strict=False)
    return model

