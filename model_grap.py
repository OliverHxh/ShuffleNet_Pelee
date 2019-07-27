import torch
from torchsummary import summary
from shuffleNet import ShuffleNet
#from torchvision.models import vgg11
from configs.CC import Config

cfg = Config.fromfile('configs/Pelee_VOC.py')
model = ShuffleNet("test", 304, cfg.model)
if torch.cuda.is_available():
    model.cuda()
summary(model, (3, 304, 304))

