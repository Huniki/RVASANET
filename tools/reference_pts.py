import numpy
import numpy as np
import torch
# coordinates_nopts = np.arange(0,320,1,dtype=int)
IMAGE_SIZE: [256, 704]
IN_CHANNEL: 256
OUT_CHANNEL: 80
FEATURE_SIZE: [32, 88]
XBOUND: [-54.0, 54.0, 0.3]
YBOUND: [-54.0, 54.0, 0.3]
ZBOUND: [-10.0, 10.0, 20.0]
DBOUND: [1.0, 60.0, 0.5]

image_size = (256,704)
feature_size = (32,88)
# dbound = [1.0, 60.0, 0.5]
iH, iW = image_size
fH, fW = feature_size

# ds = torch.arange(1.0,60.0,0.5, dtype=torch.float).view(-1, 1, 1).expand(-1, fH, fW)
# D, _, _ = ds.shape
# xs = torch.linspace(0, iW - 1, fW, dtype=torch.float).view(1, 1, fW).expand(D, fH, fW)
# ys = torch.linspace(0, iH - 1, fH, dtype=torch.float).view(1, fH, 1).expand(D, fH, fW)
# frustum = torch.stack((xs, ys, ds), -1)
# xs = torch.linspace(0, 320, fW, dtype=torch.float).view(1, 1, fW).expand(D, fH, fW)


xs = torch.arange(0, 320, 1, dtype=torch.float).view(1, 320).expand(320, 320)
ys = torch.arange(0, 320, 1, dtype=torch.float).view(320, 1).expand(320, 320)
hs = torch.zeros((320,320))
frustum = torch.stack((hs,ys,xs), -1)
frustum = frustum.reshape(-1,3)

xs = np.arange(0, 320, 1, dtype=float).reshape(1,320).repeat(320,0)
ys = np.arange(0, 320, 1, dtype=float).reshape(320,1).repeat(320,1)
hs = np.zeros((320,320))
frustum = np.stack((hs,ys,xs), -1)
frustum = frustum.reshape(-1,3)




print('debug')



# def create_frustum(self):
#     iH, iW = self.image_size
#     fH, fW = self.feature_size
#
#     ds = torch.arange(*self.dbound, dtype=torch.float).view(-1, 1, 1).expand(-1, fH, fW)
#     D, _, _ = ds.shape
#     xs = torch.linspace(0, iW - 1, fW, dtype=torch.float).view(1, 1, fW).expand(D, fH, fW)
#     ys = torch.linspace(0, iH - 1, fH, dtype=torch.float).view(1, fH, 1).expand(D, fH, fW)
#     frustum = torch.stack((xs, ys, ds), -1)
#
#     return nn.Parameter(frustum, requires_grad=False)