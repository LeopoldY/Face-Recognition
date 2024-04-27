import torch
from torch import nn
import cv2
import torchvision.transforms as T
from backbones.resnet import iresnet18


if __name__ == '__main__':
    img1 = cv2.imread('data/images/img_0.jpg') 
    img2 = cv2.imread('data/images/img_2.jpg')

    transform = T.Compose([
                    T.ToTensor(),
                    T.Resize((112, 112)),
                    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ])
    
    img1 = transform(img1)
    img2 = transform(img2)

    img1 = img1.unsqueeze(0)
    img2 = img2.unsqueeze(0)

    model = iresnet18()
    state_dict = torch.load(r'saves\checkpoint_arcface_iresnet18_33.pth')
    model.load_state_dict(state_dict['model_state_dict'])
    model.eval()

    feat1 = model(img1)
    feat2 = model(img2)

    sim = nn.functional.cosine_similarity(feat1, feat2)
    print(sim)

