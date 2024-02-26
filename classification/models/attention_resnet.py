import torch
import torch.nn as nn
import torchvision.models as models
from .attention_modules import CBAM

class AttentionResNet(nn.Module):
    def __init__(self, depth=50, attention_masks=[False, False, False, False], pretrained=True):
        '''
        depth: ResNet의 깊이. 18, 34, 50, 101, 152 중 하나
        attention_masks : 4 size attention masks [bool, bool, bool, bool] (default: [False, False, False, False]
        pretrained: ImageNet으로 학습된 모델을 로드할지 여부
        '''
        super(AttentionResNet, self).__init__()
        self.depth = depth
        self.pretrained = pretrained
        
        if self.depth not in [18, 34, 50, 101, 152]:
            raise ValueError('Invalid depth: {}'.format(self.depth))
        
        self.attention_masks = attention_masks
        
        self.resnet = getattr(models, 'resnet{}'.format(self.depth))(pretrained=self.pretrained)
        self.resnet.layer3[-1].add_module('cbam', CBAM(100))
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def load_finetuned_model(self, num_classes, path):
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes).to(self.device)
        self.load_model(path)
        
    def forward(self, x):
        return self.resnet(x)
    
    def get_feature_params(self):
        return self.resnet.parameters()
    
    def save_model(self, path):
        print('Saving model to: {}'.format(path))
        torch.save(self.resnet.state_dict(), path)
        
    def load_model(self, path):
        try:
            print('Loading model from: {}'.format(path))
            self.resnet.load_state_dict(torch.load(path))
        except:
            print('Failed to load model from: {}'.format(path))
