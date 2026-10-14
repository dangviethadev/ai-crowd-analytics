"""Crowd density estimation: VGG19 + regression head from
"Bayesian Loss for Crowd Count Estimation with Point Supervision" (Ma et al., ICCV 2019)."""
from pathlib import Path

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

VGG19_CFG = [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512]
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class VGG19Density(nn.Module):
    def __init__(self):
        super().__init__()
        layers, in_ch = [], 3
        for v in VGG19_CFG:
            if v == "M":
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                layers += [nn.Conv2d(in_ch, v, kernel_size=3, padding=1), nn.ReLU(inplace=True)]
                in_ch = v
        self.features = nn.Sequential(*layers)
        self.reg_layer = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True),
        )
        self.density_layer = nn.Sequential(nn.Conv2d(128, 1, 1), nn.ReLU())

    def forward(self, x):
        x = self.features(x)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        return self.density_layer(self.reg_layer(x))


class DensityEstimator:
    def __init__(self, weights_path, device="cpu"):
        if not Path(weights_path).exists():
            raise FileNotFoundError(f"Density model is not installed (missing {weights_path}). "
                                    "Use Detection mode, or add the file to the weights folder.")
        self.device = torch.device(device)
        self.model = VGG19Density()
        self.model.load_state_dict(torch.load(weights_path, map_location=self.device, weights_only=True))
        self.model.to(self.device).eval()

    def predict(self, frame_bgr):
        """Return (density_map, count). The map is ~1/8 of the input size and sums to the count."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        x = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255).unsqueeze(0)
        x = ((x - MEAN) / STD).to(self.device)
        with torch.no_grad():
            density = self.model(x)[0, 0].cpu().numpy()
        return density, float(density.sum())
