import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from tqdm import tqdm

from models.attention_resnet import AttentionResNet

from utils.image_transform import basic_transform

# 특정 층의 출력을 저장할 리스트


label_list = []
base_model_features = []
attention_model_features = []

def base_model_hook(module, input, output):
    base_model_features.append(output.view(output.size(0), -1).cpu().detach()) # batch, num_features, W, H -> batch, num_features*W*H


def main():
    global base_model_features
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # load baseline model
    base_model = models.resnet50(pretrained=True).to(device)
    
    # load attention model
    attention_model = AttentionResNet(depth=50, attention_masks=[False, False, False, True], pretrained=True).to(device)
    attention_model.load_finetuned_model(num_classes=10, path='model.pth')        
    attention_model.eval()
    
    # load dataset
    test_dataset = datasets.STL10(root='./datasets/stl10', split='test', transform=basic_transform)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)
    
    handle = base_model.avgpool.register_forward_hook(base_model_hook)
    
    base_model.eval()
    attention_model.eval()
    
    # for noise_type in ['']:
    #     for noise_weight in range(0, 1.0, 0.01):
    #         for image, label in test_dataloader:
    #             image = image.unsqueeze(0).to(device)
    #             base_model(image)
    for image, label in test_dataloader:
        image = image.to(device)
        base_model(image)

    base_model_features = torch.cat(base_model_features, dim=0)
    print(base_model_features.shape)
    
if __name__ == '__main__':
    main()