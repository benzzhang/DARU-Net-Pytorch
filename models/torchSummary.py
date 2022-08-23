from models.att_unet_ori import AttUNet
from models.unet_ori import UNet
from models.DARU_Net import DARU_Net
import torch
from torchsummary import summary

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
U_model = UNet().to(device)
DARU_model = DARU_Net().to(device)

summary(U_model, input_size=(3, 512, 512))
# summary(DARU_model, input_size=(1, 512, 512))
