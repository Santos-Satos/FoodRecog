import numpy as np
import torch
import random
from torch.utils.data import DataLoader
from Model import bidirectional_cnnlstm
import torch.nn as nn
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from torchinfo import summary


# モデルを学習させるプログラム
def main(model, batch_size, optimizer, epochs, progress_flag=True, save_flag=False):
    
    device = SelectDevice()

    # シード値の固定
    torch_fix_seed()

    # データセットの取得
    train_x, train_y = LoadDatasets()
    
    # データセット数の取得
    train_num = train_y.size()[0]
    
    # データセットの作成
    train_set = torch.utils.data.TensorDataset(train_x, train_y)

    # ミニバッチ取得
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

    # 損失関数の定義
    criterion = nn.CrossEntropyLoss()

    # 学習時に最適化を行うためのループを行う
    for epoch in range(epochs):
        if progress_flag == True: print("epoch:", epoch)

        # 学習
        model.train() # 学習モード
        for in_data, labels in train_loader:
            in_data.to(device)

            # 学習時の推定と評価
            pred = model(in_data)

            # パラメータの更新
            loss = criterion(pred, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
    # パラメータの保存
    if save_flag == True:
        # 学習情報の保存
        outfile = './params/state_cnnbilstm.pth'
        torch.save(model.state_dict(), outfile)
        print("save params ->", outfile)
    
    return loss

# データセットを取得する関数
def LoadDatasets():
    # データの読み込み
    train_x = np.load("./dataset/X_train.npy")
    train_y = np.load("./dataset/Y_train.npy")
    
    # one-hotベクトルに変換
    n_labels = len(np.unique(train_y))
    train_y = np.eye(n_labels)[train_y]
    
    # tensorへ変換
    train_x = torch.from_numpy(train_x.astype(np.float32)).clone()
    train_y = torch.from_numpy(train_y.astype(np.float32)).clone()
    
    # デバイスの選択
    device = SelectDevice()
    train_x = train_x.to(device)
    train_y = train_y.to(device)

    return train_x, train_y


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
    # モデルの定義
    with open("Food_class.txt", mode="r") as f:
         food_class = f.read().split("\n")[0].split(" ")
    model = bidirectional_cnnlstm(n_channel=32, n_output=len(food_class))
    device = SelectDevice()
    model = model.to(device)
    
    # ハイパーパラメータ
    epochs = 20
    batch_size = 64
    lr = 0.01
    betas = (0.9, 0.999)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=betas)
    
    main(model, batch_size, optimizer, epochs, progress_flag=True, save_flag=False)
