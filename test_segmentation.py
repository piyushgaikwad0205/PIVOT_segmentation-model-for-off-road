"""
Segmentation Test / Evaluation Script
Runs inference on dataset, evaluates metrics and saves predictions
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch import nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import cv2
import os
from tqdm import tqdm

# =========================================================
# CLASS DEFINITIONS
# =========================================================

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

class_names = [
"Background",
"Trees",
"Lush Bushes",
"Dry Grass",
"Dry Bushes",
"Ground Clutter",
"Flowers",
"Logs",
"Rocks",
"Landscape",
"Sky"
]

n_classes = len(value_map)

color_palette = np.array([
[0,0,0],
[34,139,34],
[0,255,0],
[210,180,140],
[139,90,43],
[128,128,0],
[255,0,255],
[139,69,19],
[128,128,128],
[160,82,45],
[135,206,235]
],dtype=np.uint8)

# =========================================================
# MASK UTILITIES
# =========================================================

def convert_mask(mask):

    arr = np.array(mask)
    new_arr = np.zeros_like(arr,dtype=np.uint8)

    for raw,new in value_map.items():
        new_arr[arr==raw] = new

    return Image.fromarray(new_arr)


def mask_to_color(mask):

    h,w = mask.shape
    color_mask = np.zeros((h,w,3),dtype=np.uint8)

    for cls in range(n_classes):
        color_mask[mask==cls] = color_palette[cls]

    return color_mask


# =========================================================
# LABELING FUNCTION
# =========================================================

def draw_labels(image, mask):

    overlay = image.copy()

    for class_id in range(1, n_classes):

        class_mask = (mask == class_id).astype(np.uint8)

        contours,_ = cv2.findContours(
            class_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        for cnt in contours:

            area = cv2.contourArea(cnt)

            if area < 800:
                continue

            x,y,w,h = cv2.boundingRect(cnt)

            label = class_names[class_id]

            cv2.rectangle(
                overlay,
                (x,y),
                (x+w,y+h),
                (255,255,255),
                1
            )

            (tw,th),_ = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                1
            )

            cv2.rectangle(
                overlay,
                (x,y-th-4),
                (x+tw+4,y),
                (0,0,0),
                -1
            )

            cv2.putText(
                overlay,
                label,
                (x+2,y-2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255,255,255),
                1,
                cv2.LINE_AA
            )

    return overlay


# =========================================================
# DASHBOARD VISUALIZATION
# =========================================================

def build_dashboard(camera, segmentation, perception):

    h,w,_ = camera.shape

    dashboard = np.zeros((h, w*3, 3), dtype=np.uint8)

    dashboard[:,0:w] = camera
    dashboard[:,w:w*2] = segmentation
    dashboard[:,w*2:w*3] = perception

    font = cv2.FONT_HERSHEY_SIMPLEX

    cv2.putText(dashboard,"Camera",(20,40),font,1,(255,255,255),2)
    cv2.putText(dashboard,"Segmentation",(w+20,40),font,1,(255,255,255),2)
    cv2.putText(dashboard,"Terrain Perception",(w*2+20,40),font,1,(255,255,255),2)

    return dashboard


# =========================================================
# DATASET
# =========================================================

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

        img_path = os.path.join(self.image_dir,name)
        mask_path = os.path.join(self.mask_dir,name)

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        mask = convert_mask(mask)

        if self.transform:
            image = self.transform(image)

        if self.mask_transform:
            mask = self.mask_transform(mask)*255

        return image,mask,name


# =========================================================
# MODEL
# =========================================================

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


# =========================================================
# METRICS
# =========================================================

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

    return np.nanmean(ious),ious


def compute_pixel_accuracy(pred,target):

    pred = torch.argmax(pred,dim=1)

    return (pred==target).float().mean().cpu().numpy()


# =========================================================
# MAIN
# =========================================================

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Using device:",device)

    script_dir = os.path.dirname(os.path.abspath(__file__))

    model_path = os.path.join(script_dir,"segmentation_head.pth")
    data_dir = os.path.join(script_dir,"Offroad_Segmentation_testImages")
    output_dir = os.path.join(script_dir,"predictions")

    masks_dir = os.path.join(output_dir,"masks")
    masks_color_dir = os.path.join(output_dir,"masks_color")
    overlay_dir = os.path.join(output_dir,"overlay")

    os.makedirs(masks_dir,exist_ok=True)
    os.makedirs(masks_color_dir,exist_ok=True)
    os.makedirs(overlay_dir,exist_ok=True)

    w = int(((960/2)//14)*14)
    h = int(((540/2)//14)*14)

    transform = transforms.Compose([
        transforms.Resize((h,w)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    mask_transform = transforms.Compose([
        transforms.Resize((h,w),interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor()
    ])

    dataset = MaskDataset(data_dir,transform,mask_transform)
    loader = DataLoader(dataset,batch_size=2,shuffle=False)

    print("Dataset size:",len(dataset))

    print("Loading DINOv2")

    backbone = torch.hub.load("facebookresearch/dinov2","dinov2_vits14")

    backbone.eval()
    backbone.to(device)

    dummy = torch.randn(1,3,h,w).to(device)

    with torch.no_grad():
        tokens = backbone.forward_features(dummy)["x_norm_patchtokens"]

    embed_dim = tokens.shape[2]

    model = SegmentationHeadConvNeXt(embed_dim,n_classes,w//14,h//14).to(device)

    model.load_state_dict(torch.load(model_path,map_location=device))

    model.eval()

    print("Model loaded")

    ious = []
    accs = []

    with torch.no_grad():

        pbar = tqdm(loader)

        for imgs,masks,names in pbar:

            imgs = imgs.to(device)
            masks = masks.to(device)

            tokens = backbone.forward_features(imgs)["x_norm_patchtokens"]

            logits = model(tokens)

            logits = F.interpolate(
                logits,
                size=imgs.shape[2:],
                mode="bilinear",
                align_corners=False
            )

            masks_gt = masks.squeeze(1).long()

            iou,_ = compute_iou(logits,masks_gt)
            acc = compute_pixel_accuracy(logits,masks_gt)

            ious.append(iou)
            accs.append(acc)

            preds = torch.argmax(logits,1)

            for i in range(imgs.shape[0]):

                name = names[i]
                base = os.path.splitext(name)[0]

                pred_mask = preds[i].cpu().numpy().astype(np.uint8)

                Image.fromarray(pred_mask).save(
                    os.path.join(masks_dir,f"{base}.png")
                )

                color_mask = mask_to_color(pred_mask)

                cv2.imwrite(
                    os.path.join(masks_color_dir,f"{base}.png"),
                    cv2.cvtColor(color_mask,cv2.COLOR_RGB2BGR)
                )

                img = imgs[i].cpu().numpy()

                mean = np.array([0.485,0.456,0.406])
                std = np.array([0.229,0.224,0.225])

                img = np.moveaxis(img,0,-1)
                img = img*std+mean
                img = np.clip(img,0,1)
                img = (img*255).astype(np.uint8)

                overlay = cv2.addWeighted(img,0.45,color_mask,0.65,0)
                overlay = draw_labels(overlay,pred_mask)

                dashboard = build_dashboard(img,color_mask,overlay)

                cv2.imwrite(
                    os.path.join(overlay_dir,f"{base}.png"),
                    cv2.cvtColor(dashboard,cv2.COLOR_RGB2BGR)
                )

    print("\nEvaluation Results")
    print("Mean IoU:",np.nanmean(ious))
    print("Mean Accuracy:",np.mean(accs))

    print("\nOutputs saved in predictions/")


if __name__ == "__main__":
    main()