import os
import shutil
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.model_selection import train_test_split
from PIL import Image
import csv 

# 設定使用的裝置
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 分割並標籤數據的函數
def split_and_label_data(image_folder):
    original_folder = os.path.join(image_folder, 'original')
    all_images = [f for f in os.listdir(original_folder) if f.endswith('.png')]
    
    # 文件名包含"positive"表示正例，不包含則為反例
    positive_images = [img for img in all_images if 'positive' in img]
    negative_images = [img for img in all_images if 'positive' not in img]

    # 分割數據
    train_positive, test_positive = train_test_split(positive_images, test_size=0.2, random_state=1)
    train_negative, test_negative = train_test_split(negative_images, test_size=0.2, random_state=1)

    train_images = train_positive + train_negative
    test_images = test_positive + test_negative

    # 創建訓練和測試資料夾
    train_dir = os.path.join(image_folder, 'train')
    test_dir = os.path.join(image_folder, 'test')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # 移動文件到相應文件夾
    def move_files(image_list, target_folder):
        for img in image_list:
            shutil.copy(os.path.join(original_folder, img), os.path.join(target_folder, img))
    move_files(train_images, train_dir)
    move_files(test_images, test_dir)

    # 創建標籤文件
    def create_labels_file(image_list, folder):
        labels = [{'filename': img, 'label': 1 if 'positive' in img else 0} for img in image_list]
        df = pd.DataFrame(labels)
        df.to_csv(os.path.join(folder, 'labels.csv'), index=False)
    create_labels_file(train_images, train_dir)
    create_labels_file(test_images, test_dir)

# 自定義數據集類別
class CustomImageDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.img_labels = pd.read_csv(os.path.join(img_dir, 'labels.csv'))
        self.transform = transform
        print(f"讀取到的圖片標籤數: {len(self.img_labels)}")##############

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = Image.open(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        return image, label

# 數據轉換
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# 模型定義
class BinaryClassifier(nn.Module):
    def __init__(self):
        super(BinaryClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2)
        self.fc1 = nn.Linear(64 * 56 * 56, 1024)
        self.dropout = nn.Dropout(p=0.5)  ##################### Dropout層
        self.fc2 = nn.Linear(1024, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Flatten the layer
        x = F.relu(self.fc1(x))
        x = self.dropout(x)  # 使用Dropout
        x = torch.sigmoid(self.fc2(x))
        return x

print("訓練數據總量:", len(train_dataset))##############################################
print("每個批次的大小:", train_loader.batch_size)
print("總批次數量:", len(train_loader))
# 訓練和測試函數
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device).float().unsqueeze(1)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        ############################################################
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(data), len(train_loader.dataset),
            100. * batch_idx / len(train_loader), loss.item()))
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device).float().unsqueeze(1)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.round()
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

# 主執行程式

split_and_label_data('image_folder')
train_dataset = CustomImageDataset(img_dir='image_folder/train', transform=transform)
test_dataset = CustomImageDataset(img_dir='image_folder/test', transform=transform)
train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)
model = BinaryClassifier().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.01)
criterion = nn.BCELoss()


NUM_EPOCHS = 30 # 增加訓練次數
for epoch in range(1, NUM_EPOCHS + 1):
    train(model, DEVICE, train_loader, optimizer, epoch)
    test(model, DEVICE, test_loader)


torch.save(model.state_dict(), 'model.pth')



## 未知資料測試


# 創建未知數據的標籤CSV文件
def create_unknown_labels(image_folder):
    unknown_folder = os.path.join(image_folder, 'unknown')
    all_images = [f for f in os.listdir(unknown_folder) if f.endswith('.png')]

    labels = []
    for img in all_images:
        label = 1 if 'positive' in img else 0
        labels.append({'filename': img, 'label': label})

    df = pd.DataFrame(labels)
    df.to_csv(os.path.join(unknown_folder, 'labels.csv'), index=False)

# 呼叫函數以創建未知數據的標籤CSV文件
create_unknown_labels('image_folder')

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


# 創建數據加載器
unknown_dataset = CustomImageDataset(img_dir='image_folder/unknown', transform=transform)
unknown_loader = DataLoader(dataset=unknown_dataset, batch_size=16, shuffle=False)


# 加載訓練好的模型
model = BinaryClassifier().to(DEVICE)
model.load_state_dict(torch.load('model.pth'))  # 載入已保存的模型權重
model.eval()  # 設置模型為評估模式

# 自定義數據集類別
class CustomImageDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.img_labels = pd.read_csv(os.path.join(img_dir, 'labels.csv'))
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = Image.open(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        return image, label

# 使用模型進行預測
predictions = []

with torch.no_grad():
    for images, img_paths in unknown_loader:
        images = images.to(DEVICE)
        outputs = model(images)
        predicted_labels = (outputs > 0.5).float().cpu().numpy()

        for img_path, label in zip(img_paths, predicted_labels):
            predictions.append((img_path, label))

# 將預測結果保存到CSV文件
with open('predictions.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Image', 'Prediction'])
    for img_path, label in predictions:
        writer.writerow([img_path, int(label)])

# 使用模型進行預測並計算正確率
predictions = []
correct = 0
total = 0

with torch.no_grad():
    for images, labels in unknown_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE).float().unsqueeze(1)
        outputs = model(images)
        predicted_labels = (outputs > 0.5).float()

        correct += (predicted_labels == labels).sum().item()
        total += labels.size(0)

        for img, label in zip(labels, predicted_labels):
            predictions.append((img.cpu().numpy(), label.cpu().numpy()))

accuracy = correct / total
print(f"Unknown data accuracy: {accuracy:.2%}")
