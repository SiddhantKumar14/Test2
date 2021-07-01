import os
from stn import STNet

import cv2
import torch
import numpy as np


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
STN = STNet()
STN.to(device)


STN.load_state_dict(
    torch.load("Final_STN_model.pth", map_location=lambda storage, loc: storage)
)
STN.eval()

im = cv2.imread(
    "/home/siddhant.kumar.14/siddhant-ml-ocr-training/images/0001_2021_03_02_18_03_15_76_472_249_534_278_inkh83.jpg"
)
print("im shape = ", im.shape)

im = cv2.resize(im, (94, 24), interpolation=cv2.INTER_CUBIC)
im = (np.transpose(np.float32(im), (2, 0, 1)) - 127.5) * 0.0078125
data = torch.from_numpy(im).float().unsqueeze(0).to(device)
transfer = STN(data)
transfer = transfer.cpu().detach().numpy()

print("shape = ",im.shape, transfer[0].shape)
