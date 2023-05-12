import os
import prog04_Train_model
from Model import bidirectional_cnnlstm
import torch
import optuna
import json
optuna.logging.disable_default_handler()

trial_cnt = 0

# optunaを用いてハイパーパラメータを最適化するプログラム
def main():
    if os.path.isfile("log_parameter_tuning.txt") == True:
        os.remove("log_parameter_tuning.txt")
    TRIAL_SIZE = 500
    study = optuna.create_study()
    study.optimize(objective, n_trials=TRIAL_SIZE)
    print(study.best_params)
    with open("log_parameter_tuning.txt", mode="a") as f:
        f.write(json.dumps(study.best_params))


# 最適化関数
def objective(trial):
    global trial_cnt
    device = SelectDevice()
    
    # モデルの定義
    with open("Food_class.txt", mode="r") as f:
        food_class = f.read().split("\n")[0].split(" ")
    model = bidirectional_cnnlstm(n_channel=32, n_output=len(food_class))
    model = model.to(device)
    
    optimizer = get_optimizer(trial, model)
    batch_size = trial.suggest_categorical("batch_size", [1, 2, 4, 8, 16, 32, 64])
    epochs = 20
    loss = prog04_Train_model.main(model, batch_size, optimizer, epochs, progress_flag=False, save_flag=False)
    loss = loss.item()

    print("Trial:", trial_cnt, "loss:", loss)
    with open("log_parameter_tuning.txt", mode="a") as f:
        f.write("Trial:"+str(trial_cnt)+"   loss:"+str(loss)+"\n")
    trial_cnt += 1
    
    return loss

# デバイスを取得する関数
def SelectDevice():
    return "cuda" if torch.cuda.is_available() else "cpu"

# optimizerを選択する関数
def get_optimizer(trial, model):
    optimizer_names = ["Adam", "MomentumSGD", "rmsprop"]
    optimizer_name = trial.suggest_categorical("optimizer", optimizer_names)
    weight_decay = trial.suggest_loguniform("weight_decay", 1e-10, 1e-3)

    if optimizer_name == optimizer_names[0]:
        adam_lr = trial.suggest_loguniform("adam_lr", 1e-5, 1e-1)
        optimizer = torch.optim.Adam(model.parameters(), lr=adam_lr, weight_decay=weight_decay)
    elif optimizer_name == optimizer_names[1]:
        momentum_sgd_lr = trial.suggest_loguniform("momentum_sgd_lr", 1e-5, 1e-1)
        optimizer = torch.optim.SGD(model.parameters(), lr=momentum_sgd_lr, momentum=0.9, weight_decay = weight_decay)
    else:
        optimizer = torch.optim.RMSprop(model.parameters())

    return optimizer

if __name__ == "__main__":
    main()
