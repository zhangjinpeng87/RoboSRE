import torch
import torch.optim as optim
import torch.nn as nn
from src.data_loader import ImgLoader
from src.model import ResNet

def train(data_dir, batch_size, num_workers, num_classes, pretrained, num_epochs, lr):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img_loader = ImgLoader(data_dir, batch_size, num_workers)
    data_loader = img_loader.load_data()
    model = ResNet(num_classes, pretrained).get_model()
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(num_epochs):
        model.train()
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}')
    torch.save(model.state_dict(), 'model.pth')