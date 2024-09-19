import os
import numpy as np
import cv2
import torch
import random
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor, Resize, Compose, Grayscale, ToPILImage, Normalize
import json
from datetime import datetime
import matplotlib.pyplot as plt

# 全局变量
mse_criterion = nn.MSELoss()
mae_criterion = nn.L1Loss()
metrics_history = {'train_loss': [], 'val_loss': [], 'train_tiae': [], 'train_mae': [], 'val_tiae': [], 'val_mae': []}

# 数据集路径
original_image_dir = '/home/featurize/data/已处理原图'
denoised_image_dir = '/home/featurize/data/清晰图像'

# 超参数
EPOCHS = 300
BATCH_SIZE = 6
IMAGE_SIZE = (256, 256)  # 图像大小
LEARNING_RATE = 1e-4
best_loss = 1.0


def seed_torch(seed=23): #随机数种子1029
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed_torch() 

class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()
        self.softmax = nn.Softmax(dim=1)  # 定义softmax函数

    def forward(self, A, B, C):
        # Step 1: 计算 B - A 的每个像素的绝对值之差
        abs_diff = torch.abs(B - A)

        # Step 2: 计算变化率
        change_ratio = self.softmax(abs_diff.view(1, -1)).view_as(abs_diff)
        
        # Step 3: 计算 B 和 C 的每个像素的绝对值误差
        L = torch.abs(B - C)
        
        # Step 4: 计算矩阵 H，并将矩阵 H 的每个元素值加起来，得到最终的损失
        H = change_ratio * L
        loss = H.sum()
        return loss
    
# 自定义数据集
class DenoiseDataset(Dataset):
    def __init__(self, original_dir, denoised_dir, transform=None):
        self.original_dir = original_dir
        self.denoised_dir = denoised_dir
        self.transform = transform
        self.image_names = [f for f in os.listdir(original_dir) if os.path.isfile(os.path.join(original_dir, f))]

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        original_image_path = os.path.join(self.original_dir, self.image_names[idx])
        denoised_image_path = os.path.join(self.denoised_dir, self.image_names[idx])
        
        original_image = cv2.imread(original_image_path, cv2.IMREAD_GRAYSCALE)
        denoised_image = cv2.imread(denoised_image_path, cv2.IMREAD_GRAYSCALE)
       
        original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)  # 转换为3通道
        denoised_image = cv2.cvtColor(denoised_image, cv2.COLOR_GRAY2RGB)  # 转换为3通道

        if self.transform:
            original_image = self.transform(original_image)
            denoised_image = self.transform(denoised_image)

        return original_image, denoised_image

# 数据预处理
transform = Compose([
    ToPILImage(),
    Resize(IMAGE_SIZE),
    ToTensor(),
    Grayscale(num_output_channels=1),  # 转回单通道
])

# 加载数据集
dataset = DenoiseDataset(original_image_dir, denoised_image_dir, transform=transform)
train_size = int(0.8 * len(dataset))
val_size = len(dataset)-train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

class DnCNN(nn.Module):
    def __init__(self, depth=50, image_channels=1, use_bnorm=True):
        super(DnCNN, self).__init__()
        kernel_size = 3
        padding = 1
        
        layers = []
        
        # First layer
        layers.append(nn.Conv2d(in_channels=image_channels, out_channels=128, kernel_size=kernel_size, padding=padding, bias=True))
        layers.append(nn.ReLU(inplace=True))
        
        self.head = nn.Sequential(*layers)
        
        # Intermediate layers with residual connections
        self.body = nn.ModuleList()
        for i in range((depth - 2) // 3):
            block = []
            for _ in range(3):
                block.append(nn.Conv2d(in_channels=128, out_channels=128, kernel_size=kernel_size, padding=padding, bias=False))
                if use_bnorm:
                    block.append(nn.BatchNorm2d(128))
                block.append(nn.ReLU(inplace=True))
            self.body.append(nn.Sequential(*block))
        
        # Last layer
        self.tail = nn.Conv2d(in_channels=128, out_channels=image_channels, kernel_size=kernel_size, padding=padding, bias=True)
        self.Normalize = nn.Tanh()
        
    def forward(self, x):
        out = self.head(x)
        residual = out
        for block in self.body:
            out = block(out)
            out = out + residual  # Add residual connection
            residual = out  # Update residual
        out = self.Normalize(self.tail(out))
        return out

# 初始化模型、损失函数和优化器
model = DnCNN().cuda()
print(model)

'''
# 加载模型参数
model_params = torch.load('dncnn_model_best.pth')
# 加载模型参数
model.load_state_dict(model_params)
'''
criterion = CustomLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 训练模型
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=EPOCHS, validate_every=5):
    
    global metrics_history
    
    model.train()
    
    for epoch in range(num_epochs):
        
        epoch_loss = 0
        train_tiae = 0
        train_mae = 0
        
        for data in train_loader:
            noisy_imgs, clean_imgs = data
            noisy_imgs = noisy_imgs.cuda()
            clean_imgs = clean_imgs.cuda()
        
            optimizer.zero_grad()
            outputs = model(noisy_imgs)
        
            loss = mse_criterion(outputs, clean_imgs)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
            # 计算并累加 MSE 和 MAE
            train_tiae += criterion(noisy_imgs, clean_imgs, outputs).item()
            train_mae += mae_criterion(outputs, clean_imgs).item()
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss/len(train_loader)}, {train_tiae / len(train_loader)}, {train_mae / len(train_loader)}')
 
        # 记录当前 epoch 的训练指标
        metrics_history['train_loss'].append(epoch_loss / len(train_loader))
        metrics_history['train_tiae'].append(train_tiae / len(train_loader))
        metrics_history['train_mae'].append(train_mae / len(train_loader))
    
        if (epoch + 1) % validate_every == 0:
            validate_model(model, val_loader, criterion)
    
    # 将指标保存到 JSON 文件
    filename = datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '_metrics.json'
    with open(filename, 'w') as f:
        json.dump(metrics_history, f)
        
# 验证模型
def validate_model(model, val_loader, criterion):
    model.eval()
    
    val_loss = 0
    val_tiae = 0
    val_mae = 0
    
    global metrics_history
    global best_loss
    
    with torch.no_grad():
        for data in val_loader:
            noisy_imgs, clean_imgs = data
            noisy_imgs = noisy_imgs.cuda()
            clean_imgs = clean_imgs.cuda()
            outputs = model(noisy_imgs)
            loss = mse_criterion(outputs, clean_imgs)
            val_loss += loss.item()
    
            # 计算并累加 MSE 和 MAE
            val_tiae += criterion(noisy_imgs, clean_imgs, outputs).item()
            val_mae += mae_criterion(outputs, clean_imgs).item()
    
    cur_loss = val_loss / len(val_loader)
    
    metrics_history['val_loss'].append(cur_loss)
    metrics_history['val_tiae'].append(val_tiae / len(val_loader))
    metrics_history['val_mae'].append(val_mae / len(val_loader))
    
    print(f'Validation Loss: {cur_loss}, {val_tiae / len(val_loader)}, {val_mae / len(val_loader)}')
    
    #test_model(model, val_loader, output_dir)
    
    if cur_loss<best_loss:
        # 保存模型
        torch.save(model.state_dict(), '/home/featurize/dncnn_model_best.pth')
        best_loss = cur_loss
        # 测试模型       
    model.train()  # 重新进入训练模式


def test_model(model, val_loader, output_dir):
    model.eval()
    
    cnt = 0
   
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    with torch.no_grad():
        for data in val_loader:
            noisy_imgs, clean_imgs = data
            noisy_imgs = noisy_imgs.cuda()
            clean_imgs = clean_imgs.cuda()
            outputs = model(noisy_imgs)
            cnt = cnt + 1
            
            # 显示结果
            plt.figure(figsize=(30, 10))
            plt.subplot(1, 3, 1)
            plt.imshow(noisy_imgs[0].cpu().numpy().squeeze(), cmap='gray')
            plt.title('Noisy Image')
            plt.subplot(1, 3, 2)
            plt.imshow(outputs[0].cpu().numpy().squeeze(), cmap='gray')
            plt.title('Denoised Image')
            plt.subplot(1, 3, 3)
            plt.imshow(clean_imgs[0].cpu().numpy().squeeze(), cmap='gray')
            plt.title('Clean Image')
            
            # 保存图像
            output_path = os.path.join(output_dir, f'result_{cnt}.png')
            plt.savefig(output_path)
            plt.close()
            
            if cnt > 7:
                break
                                                                      

# 执行训练和验证
train_model(model, train_loader, val_loader, criterion, optimizer, EPOCHS, validate_every=10)