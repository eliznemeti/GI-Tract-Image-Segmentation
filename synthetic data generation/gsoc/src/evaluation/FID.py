import torch
from torchmetrics.image.fid import FrechetInceptionDistance
import cv2
import glob
import os

_ = torch.manual_seed(123)


def get_generated_imgs(imgs_dir):

    img_dirs = glob.glob(os.path.join(imgs_dir,'*'))
    torch_list = []
    for img in img_dirs[:100]:
        data = cv2.imread(img, cv2.IMREAD_UNCHANGED)
        torch_list.append(data)

    generated_stack = torch.stack(torch_list)

    return generated_stack

def get_real_imgs(imgs_dir):
        
    img_dirs = glob.glob(os.path.join(imgs_dir,'*'))
    torch_list = []
    for img in img_dirs[:100]:
        data = cv2.imread(img, cv2.IMREAD_UNCHANGED)
        torch_list.append(data)

    real_stack = torch.stack(torch_list)

    return real_stack


fid = FrechetInceptionDistance(feature=64)

generated_imgs = get_generated_imgs(imgs_dir="/home/syurtseven/gsoc/scripts/results/vae_1")
real_imgs = get_real_imgs(imgs_dir="/home/syurtseven/gsoc/data/madison")

fid.update(generated_imgs, real=False)
fid.update(real_imgs, real=False)
print(fid.compute())