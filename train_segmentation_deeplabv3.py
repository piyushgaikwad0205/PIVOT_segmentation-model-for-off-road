"""
DeepLabV3 Training Script for Offroad Segmentation
"""

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from tqdm import tqdm

plt.switch_backend("Agg")

# ======================================
# CLASS DEFINITIONS
# ======================================

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

# ======================================
# MASK CONVERSION
# ======================================

def convert_mask(mask):

    arr = np.array(mask)
    new_arr = np.zeros_like(arr,dtype=np.uint8)

    for raw,new in value_map.items():
        new_arr[arr==raw] = new

    return Image.fromarray(new_arr)

# ======================================
# DATASET
# ======================================

class MaskDataset(Dataset):

    def __init__(self,data_dir,transform=None,mask_transform=None):

        self.image_dir = os.path.join(data_dir,"Color_Images")
        self.mask_dir = os.path.join(data_dir,"Segmentation")

        self.transform = transform
        self.mask_transform = mask_transform

        self.ids = os.listdir(self.image_dir)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self,idx):

        name = self.ids[idx]

        img = Image.open(
            os.path.join(self.image_dir,name)
        ).convert("RGB")

        mask = Image.open(
            os.path.join(self.mask_dir,name)
        )

        mask = convert_mask(mask)

        if self.transform:
            img = self.transform(img)

        if self.mask_transform:
            mask = self.mask_transform(mask)*255

        return img,mask

# ======================================
# METRICS
# ======================================

def compute_iou(pred,target):

    pred = torch.argmax(pred,dim=1)
    pred = pred.view(-1)
    target = target.view(-1)

    ious = []

    for c in range(n_classes):

        p = pred==c
        t = target==c

        intersection = (p & t).sum().float()
        union = (p | t).sum().float()

        if union==0:
            ious.append(float("nan"))
        else:
            ious.append((intersection/union).cpu().numpy())

    return np.nanmean(ious)


def compute_pixel_accuracy(pred,target):

    pred = torch.argmax(pred,dim=1)

    return (pred==target).float().mean().cpu().numpy()

# ======================================
# REPORT GENERATION
# ======================================

def generate_report(history,output_dir):

    os.makedirs(output_dir,exist_ok=True)

    plt.figure(figsize=(10,5))

    plt.subplot(1,2,1)
    plt.plot(history["train_loss"],label="train")
    plt.plot(history["val_loss"],label="val")
    plt.title("Loss")
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(history["val_iou"])
    plt.title("Validation IoU")

    plt.tight_layout()

    plt.savefig(os.path.join(output_dir,"training_report.png"))

    with open(os.path.join(output_dir,"metrics.txt"),"w") as f:

        f.write("TRAINING REPORT\n\n")

        for i in range(len(history["train_loss"])):

            f.write(
                f"Epoch {i+1}\n"
                f"Train Loss: {history['train_loss'][i]:.4f}\n"
                f"Val Loss: {history['val_loss'][i]:.4f}\n"
                f"Val IoU: {history['val_iou'][i]:.4f}\n"
                f"Val Acc: {history['val_acc'][i]:.4f}\n\n"
            )

# ======================================
# TRAINING
# ======================================

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Using device:",device)

    script_dir = os.path.dirname(os.path.abspath(__file__))

    train_dir = os.path.join(
        script_dir,
        "Offroad_Segmentation_Training_Dataset",
        "train"
    )

    val_dir = os.path.join(
        script_dir,
        "Offroad_Segmentation_Training_Dataset",
        "val"
    )

    output_dir = os.path.join(script_dir,"deeplab_training")

    # ======================================
    # IMAGE SIZE
    # ======================================

    w = 640
    h = 360

    transform = transforms.Compose([
        transforms.Resize((h,w)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2,0.2),
        transforms.ToTensor()
    ])

    mask_transform = transforms.Compose([
        transforms.Resize((h,w),interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor()
    ])

    trainset = MaskDataset(train_dir,transform,mask_transform)
    valset = MaskDataset(val_dir,transform,mask_transform)

    train_loader = DataLoader(trainset,batch_size=4,shuffle=True,num_workers=4)
    val_loader = DataLoader(valset,batch_size=4)

    print("Train size:",len(trainset))
    print("Val size:",len(valset))

    # ======================================
    # LOAD DEEPLABV3
    # ======================================

    model = torchvision.models.segmentation.deeplabv3_resnet50(
        weights="DEFAULT"
    )

    model.classifier[4] = nn.Conv2d(
        256,
        n_classes,
        kernel_size=1
    )

    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(),lr=1e-4)

    loss_fn = nn.CrossEntropyLoss()

    epochs = 40

    history = {
        "train_loss":[],
        "val_loss":[],
        "val_iou":[],
        "val_acc":[]
    }

    best_iou = 0

    # ======================================
    # TRAIN LOOP
    # ======================================

    for epoch in range(epochs):

        model.train()

        train_losses = []

        pbar = tqdm(train_loader)

        for imgs,masks in pbar:

            imgs = imgs.to(device)
            masks = masks.to(device).squeeze(1).long()

            outputs = model(imgs)["out"]

            loss = loss_fn(outputs,masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

            pbar.set_description(
                f"Epoch {epoch+1}/{epochs} Loss {loss.item():.4f}"
            )

        train_loss = np.mean(train_losses)

        # ==============================
        # VALIDATION
        # ==============================

        model.eval()

        val_losses = []
        ious = []
        accs = []

        with torch.no_grad():

            for imgs,masks in val_loader:

                imgs = imgs.to(device)
                masks = masks.to(device).squeeze(1).long()

                outputs = model(imgs)["out"]

                loss = loss_fn(outputs,masks)

                val_losses.append(loss.item())

                iou = compute_iou(outputs,masks)
                acc = compute_pixel_accuracy(outputs,masks)

                ious.append(iou)
                accs.append(acc)

        val_loss = np.mean(val_losses)
        val_iou = np.nanmean(ious)
        val_acc = np.mean(accs)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_iou"].append(val_iou)
        history["val_acc"].append(val_acc)

        print("\n----------------------------------")
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss:   {val_loss:.4f}")
        print(f"Val IoU:    {val_iou:.4f}")
        print(f"Val Acc:    {val_acc:.4f}")
        print("----------------------------------")

        # save best model
        if val_iou > best_iou:

            best_iou = val_iou

            torch.save(
                model.state_dict(),
                os.path.join(output_dir,"best_model.pth")
            )

    # save final model
    torch.save(
        model.state_dict(),
        os.path.join(output_dir,"final_model.pth")
    )

    generate_report(history,output_dir)

    print("\nTraining complete")

# ======================================

if __name__ == "__main__":
    main()