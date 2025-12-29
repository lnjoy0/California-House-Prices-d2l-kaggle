import pandas as pd
import torch
import pre_process
from pathlib import Path
from torch import nn
from d2l import torch as d2l
import numpy as np
import sys
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent

def get_features_labels(train_data, all_features):
    n_train = train_data.shape[0]
    train_features = torch.tensor(all_features[:n_train].values, dtype=torch.float32, device=d2l.try_gpu())
    test_features = torch.tensor(all_features[n_train:].values, dtype=torch.float32, device=d2l.try_gpu())
    train_labels = torch.tensor(
        train_data['Sold Price'].values.reshape(-1, 1), dtype=torch.float32, device=d2l.try_gpu())

    # 使用对数标签进行训练，这样mse的绝对误差不会太大，使得训练更稳定。
    # 这样训练出来的模型输出也是对数值，需要在预测时用expm1还原。
    train_labels_log = torch.log1p(train_labels) # log1p 是 log(1+x)，更稳

    return train_features, test_features, train_labels_log

def init_weights(net):
    """对网络中的所有Linear层应用Kaiming初始化"""
    for m in net.modules():
        if isinstance(m, nn.Linear):
            # 虽然 Xavier 也不错，但针对 ReLU 激活函数，Kaiming Normal (也叫 He Initialization) 通常效果更好。
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

def log_rmse(loss, predict, labels):
    rmse = torch.sqrt(loss(predict, labels)) # 这里的predict和labels已经是对数值
    return rmse.item()

def train(net, train_features, train_labels, test_features, test_labels,
          num_epochs, learning_rate, weight_decay, batch_size, fold_num=None, patience=15):
    loss = nn.MSELoss()
    train_ls, test_ls = [], []
    train_iter = d2l.load_array((train_features, train_labels), batch_size)
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate,
                                 weight_decay=weight_decay)
    
    # 引入动态学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=patience
    )

    desc = f"Fold {fold_num}" if fold_num else "Training"
    pbar = tqdm(range(num_epochs), desc=desc, file=sys.stdout, 
                ncols=100, leave=True, dynamic_ncols=False)
    
    for epoch in pbar:
        net.train() # 开启训练模式（BN 和 Dropout 必选）
        for X, y in train_iter:
            optimizer.zero_grad()
            l = loss(net(X.to(device=d2l.try_gpu())), y.to(device=d2l.try_gpu()))
            l.backward()
            optimizer.step()

        net.eval()  # 切换到评估模式
        with torch.no_grad():
            train_rmse = log_rmse(loss, net(train_features), train_labels)
            train_ls.append(train_rmse)
            if test_labels is not None:
                valid_rmse = log_rmse(loss, net(test_features), test_labels)
                test_ls.append(valid_rmse)

                # 根据验证集的 log_rmse 调整学习率
                scheduler.step(valid_rmse)

                pbar.set_postfix({'train_rmse': f'{train_rmse:.4f}', 'valid_rmse': f'{valid_rmse:.4f}'})
            else:
                pbar.set_postfix({'train_rmse': f'{train_rmse:.4f}'})
    return train_ls, test_ls

def get_k_fold_data(k, i, X, y): # 共k折，验证集是第i折
    assert k > 1
    fold_size = X.shape[0] // k # 每一折的大小是样本数除以K
    X_train, y_train = None, None
    for j in range(k):
        idx_range = slice(j * fold_size, (j + 1) * fold_size) # 每一折的索引范围
        X_part, y_part = X[idx_range, :], y[idx_range]
        if j == i:
            X_valid, y_valid = X_part, y_part # 验证集只有一折
        elif X_train is None:
            X_train, y_train = X_part, y_part # 第一折训练集
        else:
            X_train = torch.cat((X_train, X_part), dim=0) # 拼接其余折训练集
            y_train = torch.cat((y_train, y_part), dim=0)
    return X_train, y_train, X_valid, y_valid

def k_fold_validation(net, k, X_train, y_train, num_epochs, learning_rate,
           weight_decay, batch_size, patience):
    train_l_sum, valid_l_sum = 0, 0
    for i in range(k):
        print(f'\n===== 第 {i + 1}/{k} 折 =====')
        data = get_k_fold_data(k, i, X_train, y_train)
        train_ls, valid_ls = train(
            net, *data, num_epochs, learning_rate,
            weight_decay, batch_size, fold_num=i+1, patience=patience)
        train_l_sum += train_ls[-1] # 只取最后一轮损失
        valid_l_sum += valid_ls[-1]
        if i == 0:
            d2l.plot(list(range(1, num_epochs + 1)), [train_ls, valid_ls], 
                        xlabel='epoch', ylabel='log rmse', xlim=[1, num_epochs],
                        legend=['train', 'valid'], yscale='log')
        print(f'折{i + 1}，训练log rmse{float(train_ls[-1]):f}，'
              f'验证log rmse{float(valid_ls[-1]):f}')
        torch.save(net.state_dict(), BASE_DIR.joinpath(f'pth/k_fold_net_fold_{i+1}.pth'))  # 保存每折的模型参数
        # break  # 仅运行一折实验

    return train_l_sum / k, valid_l_sum / k

def final_train_and_test(net, train_features, test_features, train_labels, test_data,
                    num_epochs, learning_rate, weight_decay, batch_size):
    print('\n===== 最终训练 =====')
    train_ls, _ = train(
         net, train_features, train_labels, None, None,
         num_epochs, learning_rate, weight_decay, batch_size, fold_num=None) # 验证集和验证标签为None
    # d2l.plot(np.arange(1, num_epochs + 1),[train_ls], xlabel='epoch',
    #          ylabel='log rmse', xlim=[1, num_epochs], yscale='log')
    print(f'训练log rmse: {float(train_ls[-1]):f}')

    # 在测试集上预测
    preds_log = net(test_features).detach().cpu().numpy()
    preds = np.expm1(preds_log)  # 将对数值还原为原始房价

    test_data['Sold Price'] = pd.Series(preds.reshape(1, -1)[0]) # 添加预测结果到dataframe

    submission = pd.concat([test_data['Id'], test_data['Sold Price']], axis=1)
    submission.to_csv(BASE_DIR.joinpath('csv/submission.csv'), index=False)

if __name__ == "__main__":
    train_data = pd.read_csv(BASE_DIR.joinpath('../../data/california-house-prices/train.csv'))
    test_data = pd.read_csv(BASE_DIR.joinpath('../../data/california-house-prices/test.csv'))

    all_features, train_data_processed = pre_process.data_preprocess(train_data, test_data)
    train_features, test_features, train_labels = get_features_labels(train_data_processed, all_features)

    k, num_epochs, lr, weight_decay, batch_size, patience  = 5, 300, 0.005, 1e-3, 1024, 15
    drop1, drop2, drop3 = 0.4, 0.2, 0.1

    net = nn.Sequential(
        nn.Linear(train_features.shape[1], 1024),
        nn.BatchNorm1d(1024), # 加入Batch Normalization，缓解梯度消失，放在线性层后，ReLU前
        nn.ReLU(),
        nn.Dropout(drop1), # 增加 Dropout 抵抗过拟合
        
        nn.Linear(1024, 512),
        nn.BatchNorm1d(512), # 再次加入 BN
        nn.ReLU(),
        nn.Dropout(drop2),

        nn.Linear(512, 256),
        nn.BatchNorm1d(256), # 再次加入 BN
        nn.ReLU(),
        nn.Dropout(drop3),
        
        nn.Linear(256, 1)
    ).to(device=d2l.try_gpu())

    print('超参数：')
    print('网络结构: ', net)
    print(f"折数: {k}, 训练轮数: {num_epochs}, 学习率: {lr}, 权重衰减: {weight_decay}, 批量大小: {batch_size}，耐心值: {patience}")
    print(f"Dropout比例: {drop1}, {drop2}, {drop3}")

    # 权重初始化
    init_weights(net)

    train_l, valid_l = k_fold_validation(net, k, train_features, train_labels,
                            num_epochs, lr, weight_decay, batch_size, patience)
    
    print(f'{k}折验证: 平均训练log rmse: {float(train_l):f}, '
        f'平均验证log rmse: {float(valid_l):f}')

    final_train_and_test(net, train_features, test_features, train_labels, test_data,
                        num_epochs, lr, weight_decay, batch_size)