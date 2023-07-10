import numpy as np
import torch
import random
from torch.utils.data import DataLoader
from Model import ResNet_Linear
import torch.nn as nn
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
from torchinfo import summary


with open("Food_class.txt", mode="r") as f:
         food_class = f.read().split("\n")[0].split(" ")

# モデルを評価するプログラム
def main(model, test_loader):
   
    # 評価
    model.eval()
    with torch.no_grad():
        preds = []
        labels = []
        equal_sum = 0
        for in_data, label in test_loader:
            pred = model(in_data)
            preds += torch.argmax(pred, dim=1).tolist()
            labels += torch.argmax(label, dim=1).tolist()
            equal_sum += (pred.argmax(dim=1)==label.argmax(dim=1)).sum()

    # 混同行列の作成
    cm = confusion_matrix(labels, preds)
    acc = (equal_sum / len(labels)).item()
    print(food_class)
    print(cm)
    print("Accuracy=",acc)
    

# データセットを取得する関数
def LoadDatasets():
    # データの読み込み
    test_x = np.load("./dataset/X_test.npy")
    test_y = np.load("./dataset/Y_test.npy")

    # one-hotベクトルに変換
    n_labels = len(np.unique(test_y))
    test_y = np.eye(n_labels)[test_y]

    # tensorへ変換
    test_x = torch.from_numpy(test_x.astype(np.float32)).clone()
    test_y = torch.from_numpy(test_y.astype(np.float32)).clone()

    # デバイスの選択
    device = SelectDevice()
    test_x = test_x.to(device)
    test_y = test_y.to(device)

    return test_x, test_y


# デバイスの取得をする関数
def SelectDevice():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# シード値を固定する関数
def torch_fix_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True

if __name__ == "__main__":
    # デバイスの選択
    device = SelectDevice()
    
    # シード値の固定
    torch_fix_seed()    

    # データセットの取得
    test_x, test_y = LoadDatasets()
    
    # データセットの作成
    test_set = torch.utils.data.TensorDataset(test_x, test_y)

    # ミニバッチ取得
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False) 

    # モデルの定義
    n_channel = 64
    model = ResNet_Linear(n_channel=64, n_output=len(food_class), drop_rate=1)
    model.load_state_dict(torch.load("./params/state_ResNet_Linear.pth"))
    model = model.to(device)
    
    main(model, test_loader)
