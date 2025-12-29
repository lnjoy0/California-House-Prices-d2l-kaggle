from autogluon.tabular import TabularPredictor, TabularDataset
import pandas as pd
from pathlib import Path
import numpy as np

BASE_DIR = Path(__file__).resolve().parent

if __name__ == '__main__':
    train_data = pd.read_csv(BASE_DIR.joinpath('../../data/california-house-prices/train.csv'))
    test_data = pd.read_csv(BASE_DIR.joinpath('../../data/california-house-prices/test.csv'))

    id, label = 'Id', 'Sold Price'

    large_val_cols = ['Lot', 'Total interior livable area',
                      'Tax assessed value', 'Last Sold Price',
                      'Listed Price', 'Annual tax amount']

    # 数据预处理，对数化大数值特征和标签
    train_data[label] = np.log1p(train_data[label])
    for c in large_val_cols:
        train_data[c] = np.log1p(train_data[c])
        test_data[c] = np.log1p(test_data[c])
    
    # 训练
    predictor = TabularPredictor(label=label).fit(
        train_data=TabularDataset(train_data).drop(columns=[id]),
        num_gpus=1,
        hyperparameters='multimodal', # 使用支持多模态的模型，其中可以使用transformer处理文本特征
        num_stack_levels=1, num_bag_folds=5
    )

    # 预测
    preds = predictor.predict(TabularDataset(test_data))
    preds_prices = np.expm1(preds)  # 将对数值还原为原始房价

    submission = pd.concat([test_data[id], preds_prices.rename(label)], axis=1)
    submission.to_csv(BASE_DIR.joinpath('predict_submission_automl.csv'), index=False)
