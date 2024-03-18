import os
import torch

import torch.utils.data as data
from torchvision.io import read_image
from torchvision.ops.boxes import masks_to_boxes
from PIL import Image
from torchvision.transforms import v2 as T
from torchvision.transforms import ToTensor


class PennFudanDataset(data.Dataset):
    def __init__(self, root: str, transforms):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))

    def __getitem__(self, index: int):
        img_path = os.path.join(self.root, "PNGImages", self.imgs[index])
        mask_path = os.path.join(self.root, "PedMasks", self.masks[index])
        img, mask = read_image(img_path), read_image(mask_path)

        # instances are encoded as different colors
        obj_ids = torch.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]
        num_objs = len(obj_ids)

        # split the color-encoded mask into a set of binary masks
        masks = (mask == obj_ids[:, None, None].to(dtype=torch.uint8))
        # get bounding box coordinates for each mask
        boxes = masks_to_boxes(masks)

        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)

        image_id = index
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs), dtype=torch.int64)

        # wrap sample and targets into torchvision tv_tensor:
        # img = Image.fromarray(img.numpy())
        # img = ToTensor(img)

        target = {}
        target["boxes"] = boxes
        target["masks"] = masks
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)


def get_transform(train):
    transforms = []
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    transforms.append(T.ToDtype(torch.float))
    transforms.append(T.ToImageTensor())
    return T.Compose(transforms)


def collate_fn(batch):
    return tuple(zip(*batch))


if __name__ == '__main__':
    from torch.utils.data import DataLoader

    dataset = PennFudanDataset(
        root="./data/PennFudanPed",
        transforms=get_transform(train=True)
    )
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn
    )
    # For Training
    images, targets = next(iter(data_loader))
    print(images.shape)
