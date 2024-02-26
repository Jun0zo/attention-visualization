import numpy as np
import torchvision.transforms as transforms
from PIL import Image, ImageEnhance, ImageFilter
import torch
import torchvision.transforms as transforms

basic_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def add_dust(img_tensor, dust_density=0.01):
    dust_mask = torch.rand_like(img_tensor) < dust_density
    dust = torch.randn_like(img_tensor) * 0.5  # 조절 가능한 dust 강도
    img_tensor[dust_mask] += dust[dust_mask]
    return torch.clamp(img_tensor, 0, 1)

def add_lighting(img_tensor, light_density=1.2):
    lighting = img_tensor * light_density
    return lighting

def add_rotation(img_tensor, max_angle=10):
    angle = np.random.uniform(-max_angle, max_angle)
    img_pil = transforms.ToPILImage()(img_tensor)
    rotated_img_pil = img_pil.rotate(angle)
    return transforms.ToTensor()(rotated_img_pil)

def add_blur(img_tensor, blur_radius=2):
    img_pil = transforms.ToPILImage()(img_tensor)
    blurred_img_pil = img_pil.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    return transforms.ToTensor()(blurred_img_pil)

class ImageNoiseApplier:
    def __init__(self, noise_density, 
                 noise_type, 
                 resize=256, 
                 center_crop_size=224, 
                 mean=[0.485, 0.456, 0.406], 
                 std=[0.229, 0.224, 0.225]):
        
        self.noise_density = noise_density
        self.noise_type = noise_type
    
        self.data_transform = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(resize, scale=(0.5, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]),
            'val': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(resize),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]),
            
            'light': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(resize),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: add_lighting(x, light_density=self.light_density)),
                transforms.Normalize(mean, std),
            ]),
            'dust': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(resize),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: add_dust(x, dust_density=self.dust_density)),
                transforms.Normalize(mean, std)
                
            ]),
            'rotate': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(resize),
                transforms.RandomRotation((self.rotate_angle, self.rotate_angle+1)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]),
            'blur': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(resize),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: add_blur(x, blur_radius=self.blur_radius)),
                transforms.Normalize(mean, std),
            ])
        }
        
    def __call__(self, img, phase):
        return self.data_transform[phase](img)