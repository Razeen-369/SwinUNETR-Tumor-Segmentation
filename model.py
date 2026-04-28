import torch
import numpy as np
from monai.networks.nets import SwinUNETR
from monai.transforms import LoadImage, EnsureChannelFirst, ScaleIntensity, ResizeWithPadOrCrop, Compose

IMG_SIZE = 224
DEVICE = "cpu"

model = SwinUNETR(
    img_size=(224, 224),
    in_channels=3,
    out_channels=1,
    feature_size=48,
    use_checkpoint=True,
    spatial_dims=2
).to(DEVICE)

model.load_state_dict(torch.load("best_swin_model.pth", map_location=DEVICE))
model.eval()

infer_img = Compose([
    LoadImage(image_only=True),
    EnsureChannelFirst(),
    ScaleIntensity(),
    ResizeWithPadOrCrop((IMG_SIZE, IMG_SIZE)),
])

def predict(image_path):
    img = infer_img(image_path)

    # FIX MetaTensor → numpy
    img = img.numpy()

    img_tensor = torch.from_numpy(img).float().unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(img_tensor)
        pred_mask = (torch.sigmoid(output) > 0.5).float().cpu().squeeze().numpy()

    return img, pred_mask