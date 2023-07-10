import numpy as np
import torch
import random
from torch.utils.data import DataLoader
from Model import ResNet_LSTM
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
    train_x, train_y, test_x, test_y = LoadDatasets()
    test_input = torch.zeros(1, train_x.size()[1], train_x.size()[2]).to(device)
    out_shape = model(test_input).size()
    train_y = CvtLabel(train_y, out_shape).to(device)
    test_y = CvtLabel(test_y, out_shape).to(device)
    
    # データセット数の取得
    train_num = train_y.size()[0]
    
    # データセットの作成
    train_set = torch.utils.data.TensorDataset(train_x, train_y)
    test_set = torch.utils.data.TensorDataset(test_x, test_y)

    # ミニバッチ取得
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    # 損失関数の定義
    criterion = nn.CrossEntropyLoss()

    # 学習時に最適化を行うためのループを行う
    max_acc = 0
    for epoch in range(epochs):
        # 学習
        model.train() # 学習モード
        equal_sum = 0
        for in_data, labels in train_loader:
            # 学習時の推定と評価
            pred = model(in_data)
            equal_sum += (torch.flatten(pred.argmax(dim=2))==torch.flatten(labels.argmax(dim=2))).sum()

            # パラメータの更新
            loss = criterion(pred, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        # 学習データの正解率算出
        train_acc = (equal_sum / (train_y.size()[0]*train_y.size()[1])).item()

        # 評価
        model.eval() # 評価モード
        with torch.no_grad():
            equal_sum = 0
            for in_data, labels in test_loader:
                pred = model(in_data)
                equal_sum += (torch.flatten(pred.argmax(dim=2))==torch.flatten(labels.argmax(dim=2))).sum()

        # 評価データの正解率算出
        test_acc = (equal_sum / (test_y.size()[0]*test_y.size()[1])).item()
        
        if progress_flag == True: 
            print("epoch:", epoch)
            print("train_acc:", train_acc, "test_acc:", test_acc)
    
        # パラメータの保存
        if save_flag == True and test_acc > max_acc:
            # 学習情報の保存
            outfile = './params/state_ResNet_LSTM.pth'
            torch.save(model.state_dict(), outfile)
            print("save params ->", outfile)
            max_acc = test_acc
                       
    return test_acc

# データセットを取得する関数
def LoadDatasets():
    # データの読み込み
    train_x = np.load("./dataset/X_train.npy")
    train_y = np.load("./dataset/Y_train.npy")
    test_x = np.load("./dataset/X_test.npy")
    test_y = np.load("./dataset/Y_test.npy")

    # one-hotベクトルに変換
    n_labels = len(np.unique(train_y))
    train_y = np.eye(n_labels)[train_y]
    test_y = np.eye(n_labels)[test_y]

    # tensorへ変換
    train_x = torch.from_numpy(train_x.astype(np.float32)).clone()
    train_y = torch.from_numpy(train_y.astype(np.float32)).clone()
    test_x = torch.from_numpy(test_x.astype(np.float32)).clone()
    test_y = torch.from_numpy(test_y.astype(np.float32)).clone()

    # デバイスの選択
    device = SelectDevice()
    train_x = train_x.to(device)
    train_y = train_y.to(device)
    test_x = test_x.to(device)
    test_y = test_y.to(device)

    return train_x, train_y, test_x, test_y

# LSTMの時系列出力のラベルに変換する関数
def CvtLabel(label, out_shape):
    cvtd_label = []
    for batch in range(label.size()[0]):
        cvtd_label.append([label[batch].tolist()]*out_shape[1])
    cvtd_label = torch.tensor(cvtd_label)
    return cvtd_label

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
    drop_rate = 0.25
    n_channel = 64

    # モデルの定義
    with open("Food_class.txt", mode="r") as f:
         food_class = f.read().split("\n")[0].split(" ")
    model = ResNet_LSTM(n_channel=n_channel, n_output=len(food_class), drop_rate=drop_rate)
    device = SelectDevice()
    model = model.to(device)
    
    # ハイパーパラメータ
    epochs = 100
    batch_size = 8
    lr = 0.01
    betas = (0.9, 0.999)
    weight_decay = 0.1
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
    
    main(model, batch_size, optimizer, epochs, progress_flag=True, save_flag=True)
