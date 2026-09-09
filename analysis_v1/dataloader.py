import pandas as pd
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset


def tensor_gen(mode: int):
    df=pd.read_csv("simplified_data.csv", index_col=0)
    # Sample 80% for df_1
    df_1 = df.sample(frac=0.8, random_state=42)
    # Select remaining 20% for df_2
    df_2 = df.drop(df_1.index)
    if mode == 0:
        df_1_x = torch.tensor(df_1.iloc[:,:16].values,dtype = torch.float32)
        df_1_y_1 = torch.tensor(df_1.iloc[:, 16].values, dtype = torch.int64)
        df_1_y_2 = torch.tensor(df_1.iloc[:, 17].values, dtype = torch.float32)
        return TensorDataset(df_1_x, df_1_y_1, df_1_y_2)
    elif mode == 1:
        df_1_x = torch.tensor(df_2.iloc[:, :16].values, dtype=torch.float32)
        df_1_y_1 = torch.tensor(df_2.iloc[:, 16].values, dtype=torch.int64)
        df_1_y_2 = torch.tensor(df_2.iloc[:, 17].values, dtype=torch.float32)
        return TensorDataset(df_1_x, df_1_y_1, df_1_y_2)

if __name__ == "__main__":
    train_dataset = tensor_gen(0)
    test_dataset = tensor_gen(1)
    print(train_dataset)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=0,  # 0 is safe on Windows
        drop_last=False
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=0
    )
    # Debug
    # In main.py, after train_dataset = tensor_gen(0)
    y_action_all = torch.cat([batch[1] for batch in DataLoader(train_dataset, batch_size=1024)])
    print("Class distribution (train):", torch.bincount(y_action_all))
