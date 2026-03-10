"""
Segmentation Training Script
Trains a segmentation head on top of DINOv2 backbone
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch import nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import cv2
from tqdm import tqdm
from torch.amp import autocast, GradScaler


# ======================================================================
# Mask Conversion
# ======================================================================

value_map = {
0:0,
100:1,
200:2,
300:3,
500:4,
550:5,
600:6,
700:7,
800:8,
7100:9,
10000:10
}

n_classes = len(value_map)

def convert_mask(mask):

    arr = np.array(mask)
    new_arr = np.zeros_like(arr,dtype=np.uint8)

    for raw_value,new_value in value_map.items():
        new_arr[arr==raw_value] = new_value

    return Image.fromarray(new_arr)


# ======================================================================
# Dataset
# ======================================================================

class MaskDataset(Dataset):

    def __init__(self,data_dir,transform=None,mask_transform=None):

        self.image_dir = os.path.join(data_dir,"Color_Images")
        self.mask_dir = os.path.join(data_dir,"Segmentation")

        self.transform = transform
        self.mask_transform = mask_transform

        self.data_ids = os.listdir(self.image_dir)

    def __len__(self):
        return len(self.data_ids)

    def __getitem__(self,idx):

        name = self.data_ids[idx]

        img_path = os.path.join(self.image_dir,name)
        mask_path = os.path.join(self.mask_dir,name)

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        mask = convert_mask(mask)

        if self.transform:
            image = self.transform(image)

        if self.mask_transform:
            mask = self.mask_transform(mask)*255

        return image,mask


# ======================================================================
# Segmentation Head
# ======================================================================

class SegmentationHeadConvNeXt(nn.Module):

    def __init__(self,in_channels,out_channels,tokenW,tokenH):

        super().__init__()

        self.H = tokenH
        self.W = tokenW

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels,128,7,padding=3),
            nn.GELU()
        )

        self.block = nn.Sequential(
            nn.Conv2d(128,128,7,padding=3,groups=128),
            nn.GELU(),
            nn.Conv2d(128,128,1),
            nn.GELU()
        )

        self.classifier = nn.Conv2d(128,out_channels,1)

    def forward(self,x):

        B,N,C = x.shape

        x = x.reshape(B,self.H,self.W,C).permute(0,3,1,2)

        x = self.stem(x)
        x = self.block(x)

        return self.classifier(x)


# ======================================================================
# Metrics
# ======================================================================

def compute_iou(pred,target,num_classes=n_classes):

    pred = torch.argmax(pred,dim=1)
    pred = pred.view(-1)
    target = target.view(-1)

    ious = []

    for c in range(num_classes):

        pred_inds = pred==c
        target_inds = target==c

        intersection = (pred_inds & target_inds).sum().float()
        union = (pred_inds | target_inds).sum().float()

        if union==0:
            ious.append(float("nan"))
        else:
            ious.append((intersection/union).cpu().numpy())

    return np.nanmean(ious)


def compute_pixel_accuracy(pred,target):

    pred = torch.argmax(pred,dim=1)

    return (pred==target).float().mean().cpu().numpy()


# ======================================================================
# Training
# ======================================================================

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Using device:",device)

    batch_size = 2
    lr = 1e-4
    n_epochs = 20

    w = int(((960/2)//14)*14)
    h = int(((540/2)//14)*14)

    script_dir = os.path.dirname(os.path.abspath(__file__))

    train_dir = os.path.join(script_dir,"Offroad_Segmentation_Training_Dataset","train")
    val_dir = os.path.join(script_dir,"Offroad_Segmentation_Training_Dataset","val")

    transform = transforms.Compose([
        transforms.Resize((h,w)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2,contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    mask_transform = transforms.Compose([
        transforms.Resize((h,w),interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor()
    ])

    trainset = MaskDataset(train_dir,transform,mask_transform)
    valset = MaskDataset(val_dir,transform,mask_transform)

    train_loader = DataLoader(trainset,batch_size=batch_size,shuffle=True,
                              num_workers=2,pin_memory=True)

    val_loader = DataLoader(valset,batch_size=batch_size,shuffle=False,
                            num_workers=2,pin_memory=True)

    print("Training samples:",len(trainset))
    print("Validation samples:",len(valset))

    print("Loading DINOv2 backbone...")

    backbone = torch.hub.load("facebookresearch/dinov2","dinov2_vits14")

    backbone.eval()
    backbone.to(device)

    print("Backbone loaded.")

    imgs,_ = next(iter(train_loader))
    imgs = imgs.to(device)

    with torch.no_grad():
        tokens = backbone.forward_features(imgs)["x_norm_patchtokens"]

    embed_dim = tokens.shape[2]

    classifier = SegmentationHeadConvNeXt(
        embed_dim,
        n_classes,
        w//14,
        h//14
    ).to(device)

    class_weights = torch.tensor([
        0.5,2.0,1.8,1.0,2.5,1.8,2.0,6.0,3.0,0.6,0.4
    ]).to(device)

    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(classifier.parameters(),lr=lr)

    scaler = GradScaler(device="cuda")

    print("\nStarting training")

    for epoch in range(n_epochs):

        classifier.train()
        train_losses=[]

        train_pbar = tqdm(train_loader,desc=f"Epoch {epoch+1}/{n_epochs} [Train]")

        for imgs,labels in train_pbar:

            imgs = imgs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            with torch.no_grad():
                tokens = backbone.forward_features(imgs)["x_norm_patchtokens"]

            with autocast("cuda"):

                logits = classifier(tokens)

                outputs = F.interpolate(
                    logits,
                    size=imgs.shape[2:],
                    mode="bilinear",
                    align_corners=False
                )

                labels = labels.squeeze(1).long()

                loss = loss_fn(outputs,labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_losses.append(loss.item())

            train_pbar.set_postfix(loss=f"{loss.item():.4f}")

        # ==============================
        # Validation
        # ==============================

        classifier.eval()

        val_losses=[]
        val_ious=[]
        val_accs=[]

        with torch.no_grad():

            for imgs,labels in tqdm(val_loader,desc=f"Epoch {epoch+1}/{n_epochs} [Val]"):

                imgs = imgs.to(device)
                labels = labels.to(device)

                tokens = backbone.forward_features(imgs)["x_norm_patchtokens"]

                logits = classifier(tokens)

                outputs = F.interpolate(
                    logits,
                    size=imgs.shape[2:],
                    mode="bilinear",
                    align_corners=False
                )

                labels = labels.squeeze(1).long()

                loss = loss_fn(outputs,labels)

                val_losses.append(loss.item())

                val_ious.append(compute_iou(outputs,labels))
                val_accs.append(compute_pixel_accuracy(outputs,labels))

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        val_iou = np.mean(val_ious)
        val_acc = np.mean(val_accs)

        print("\n----------------------------------")
        print(f"Epoch {epoch+1}/{n_epochs}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss:   {val_loss:.4f}")
        print(f"Val IoU:    {val_iou:.4f}")
        print(f"Val Acc:    {val_acc:.4f}")
        print("----------------------------------")

    model_path = os.path.join(script_dir,"segmentation_head.pth")

    torch.save(classifier.state_dict(),model_path)

    print("\nModel saved:",model_path)


if __name__ == "__main__":
    main()