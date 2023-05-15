import numpy as np
import torch
import random
from torch.utils.data import DataLoader
from Model import bidirectional_cnnlstm
import torch.nn as nn
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
from torchinfo import summary


# モデルを評価するプログラム
def main():
       
    # シード値の固定
    torch_fix_seed()
    
    # データセットの取得
    test_x, test_y = LoadDatasets()
    
    # データセット数の取得
    test_num = test_y.size()[0]

    # データセットの作成
    test_set = torch.utils.data.TensorDataset(test_x, test_y)

    # ミニバッチ取得
    batch_size = 1
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)

    # モデルの定義
    with open("Food_class.txt", mode="r") as f:
         food_class = f.read().split("\n")[0].split(" ")
    model = bidirectional_cnnlstm(n_channel=32, n_output=len(food_class))
    model.load_state_dict(torch.load("./params/state_cnnbilstm.pth"))
    model = model.to(device)

    # 評価
    model.eval()
    preds = []
    labels = []
    for in_data, label in test_loader:
        pred = model(in_data)
        preds += torch.argmax(pred, dim=1).tolist()
        labels += torch.argmax(label, dim=1).tolist()
    
    # 混同行列の作成
    cm = confusion_matrix(labels, preds)
    acc = accuracy_score(labels, preds)
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
    main()
