import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

class ImgLoader:
    def __init__(self, data_dir, batch_size, num_workers):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.dataset = ImageFolder(self.data_dir, self.transform)

    def load_data(self):
        data_loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        return data_loader