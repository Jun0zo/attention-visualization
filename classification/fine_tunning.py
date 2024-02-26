import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.models as models
from sklearn.model_selection import StratifiedShuffleSplit
from tqdm import tqdm


from models.attention_resnet import AttentionResNet

from utils.image_transform import basic_transform


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = AttentionResNet(depth=50, attention_masks=[True, True, True, True], pretrained=True)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.get_feature_params(), lr=0.001, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    train_dataset = datasets.STL10(root='./datasets/stl10', split='train', download=True, transform=basic_transform)
    train_dataset, val_dataset = train_dataset.split(0.8, stratified=True)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)
    
    test_dataset = datasets.STL10(root='./datasets/stl10', split='test', download=True, transform=basic_transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)
    
    for _ in range(5):
        model.train()
        
        for inputs, labels in tqdm(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
        scheduler.step()
    

if __name__ == "__main__":
    main()