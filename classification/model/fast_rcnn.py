import torchvision
from torch import nn, Tensor
# from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from collections import OrderedDict


# from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


class FastRCNNPredictor(nn.Module):
    def __init__(self, in_channels, num_classes):
        """
        Standard classification + bounding box regression layers for Fast R-CNN
        Args:
            in_channels (int): number of input channels
            num_classes (int): number of output classes (including background)
        """
        super(FastRCNNPredictor, self).__init__()
        self.cls_score = nn.Linear(in_channels, num_classes)
        self.bbox_pred = nn.Linear(in_channels, num_classes * 4)

    def forward(self, x: Tensor):
        if x.dim() == 4:
            assert (list(x.shape[2:]) == [1, 1]), f"x has the wrong shape, expecting the last two dimensions" \
                                                  f" to be [1,1] instead of {list(x.shape[2:])}"
        x = x.flatten(start_dim=1)
        scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)

        return scores, bbox_deltas


class MaskRCNNPredictor(nn.Sequential):
    def __init__(self, in_channels, dim_reduced, num_classes):
        super().__init__(
            OrderedDict(
                [
                    ("conv5_mask", nn.ConvTranspose2d(in_channels, dim_reduced, 2, 2, 0)),
                    ("relu", nn.ReLU(inplace=True)),
                    ("mask_fcn_logits", nn.Conv2d(dim_reduced, num_classes, 1, 1, 0)),
                ]
            )
        )

        for name, param in self.named_parameters():
            if "weight" in name:
                nn.init.kaiming_normal_(param, mode="fan_out", nonlinearity="relu")
            # elif "bias" in name:
            #     nn.init.constant_(param, 0)


def get_model_instance_segmentation(num_classes):
    """
    compute the instance segmentation mask
    Args:
        num_classes: number of the classes

    Returns:
        MaskRCNN
    """
    # load a instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask,
        hidden_layer,
        num_classes
    )

    return model


if __name__ == '__main__':
    def model_forward():
        import torch
        from data.penn_fundan import PennFudanDataset, get_transform, collate_fn

        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
        dataset = PennFudanDataset('../data/data/PennFudanPed', get_transform(train=True))
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=2,
            shuffle=True,
            num_workers=4,
            collate_fn=collate_fn
        )

        # For Training
        images, targets = next(iter(data_loader))
        images = list(image for image in images)
        targets = [{k: v for k, v in t.items()} for t in targets]
        output = model(images, targets)  # Returns losses and detections
        print(output)

        # For inference
        model.eval()
        x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
        predictions = model(x)  # Returns predictions
        print(predictions[0])


    model_forward()
