import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
import plotly.graph_objs as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from models import LSTM


features = ['epoch',
            'x_sim', 'y_sim', 'z_sim', 'Vx_sim', 'Vy_sim', 'Vz_sim',
            'dx_sim', 'dy_sim', 'dz_sim']
targets = ['x', 'y', 'z', 'Vx', 'Vy', 'Vz']


def smape(satellite_predicted_values, satellite_true_values):
    # Score function
    return torch.mean(torch.abs((satellite_predicted_values - satellite_true_values)
                          / (torch.abs(satellite_predicted_values) + torch.abs(satellite_true_values))), axis=1)


class TrainTestSequenceDataset(Dataset):
    def __init__(self, data_train: pd.DataFrame, data_test: pd.DataFrame, seq_len=2):
        self.len = len(data_train) - (seq_len - 1)
        self.seq_len = seq_len
        self.x = torch.from_numpy(data_train.values).float()
        self.y = torch.from_numpy(data_test.values).float()

    def __len__(self):
        return self.len

    def __getitem__(self, item):
        return self.x[item:item + self.seq_len], self.y[item:item + self.seq_len]


class PredictSequenceDataset(Dataset):
    def __init__(self, data: pd.DataFrame, seq_len=20):
        self.len = len(data) - (seq_len - 1)
        self.seq_len = seq_len
        self.x = torch.from_numpy(data.values).float()

    def __len__(self):
        return self.len

    def __getitem__(self, item):
        return self.x[item:item + self.seq_len]



def process_for_train_test(data):
    data = data.drop(['id'], axis=1)
    # Convert date and time to seconds
    data['epoch'] = data['epoch'].apply(lambda x: x.to_pydatetime().timestamp())
    data['epoch'] = (data['epoch'] - data['epoch'].min())

    # generate delta features
    dt = data['epoch'].values[1] - data['epoch'].values[0]
    data[['dx_sim', 'dy_sim', 'dz_sim',
          ]] = dt * data[['Vx_sim', 'Vy_sim', 'Vz_sim',
            ]]

    # Scale features
    data_to_scale = data.drop(['sat_id', 'x', 'y', 'z', 'Vx', 'Vy', 'Vz'], axis=1)
    scaler = StandardScaler()
    scaler.fit(data_to_scale)
    data_scaled = scaler.transform(data_to_scale)
    data[features] = data_scaled
    # Split by satellite id
    sat_datas = []
    data_grouped = data.groupby('sat_id')
    for sat_data in data_grouped:
        sat_datas.append(sat_data[1].drop(['sat_id'], axis=1))
    sat_datas_train = []
    sat_datas_test = []
    for sat_data in sat_datas:
        # Split data to train and test datasets
        sat_data_train, sat_data_test = train_test_split(sat_data, shuffle=False)
        sat_datas_train.append(sat_data_train)
        sat_datas_test.append(sat_data_test)
    return sat_datas_train, sat_datas_test


def process_for_predict(data):
    # Convert coordinates
    # Convert date and time to seconds
    data['epoch'] = data['epoch'].apply(lambda x: x.to_pydatetime().timestamp())
    data['epoch'] = (data['epoch'] - data['epoch'].min())
    # generate delta features
    dt = data['epoch'].values[1] - data['epoch'].values[0]
    data[['dx_sim', 'dy_sim', 'dz_sim',
          ]] = dt * data[['Vx_sim', 'Vy_sim', 'Vz_sim',
                                                            ]]
    # Scale features
    data_to_scale = data.drop(['id', 'sat_id'], axis=1)
    scaler = StandardScaler()
    scaler.fit(data_to_scale)
    data_scaled = scaler.transform(data_to_scale)
    data[features] = data_scaled
    # Split by satellite id
    sat_datas = []
    data_grouped = data.groupby('sat_id')
    for sat_data in data_grouped:
        sat_datas.append(sat_data[1])
    return sat_datas


def process_for_train(data):
    data = data.drop(['id'], axis=1)
    # Convert date and time to seconds
    data['epoch'] = data['epoch'].apply(lambda x: x.to_pydatetime().timestamp())
    data['epoch'] = (data['epoch'] - data['epoch'].min())

    # generate delta features
    dt = data['epoch'].values[1] - data['epoch'].values[0]
    data[['dx_sim', 'dy_sim', 'dz_sim',
          ]] = dt * data[['Vx_sim', 'Vy_sim', 'Vz_sim',
                                                            ]]

    # Scale features
    data_to_scale = data.drop(['sat_id'] + targets, axis=1)
    scaler = StandardScaler()
    scaler.fit(data_to_scale)
    data_scaled = scaler.transform(data_to_scale)
    data[features] = data_scaled
    # Split by satellite id
    sat_datas = []
    data_grouped = data.groupby('sat_id')
    for sat_data in data_grouped:
        sat_datas.append(sat_data[1].drop(['sat_id'], axis=1))
    return sat_datas


def make_prediction(data, sat_id):
    # model = torch.load(f'models//{sat_id}//model.pt')
    prediction_dataset = PredictSequenceDataset(data, seq_len=20)
    prediction_dataloader = DataLoader(prediction_dataset, batch_size=1, shuffle=False)
    predictions = torch.zeros(len(data), 6)
    with torch.no_grad():
        for i in range(len(data) - 12, 0, -1):
            seq_x = prediction_dataset[i]
            seq_x = seq_x.unsqueeze(0)
            # predicted_seq = model(seq_x)
            predicted_seq = torch.rand(6)
            predictions[-i - len(data):-i, :] = predicted_seq
    print('--------------------------')
    print(f'Predictions len: {len(data)}')
    print(f'Real len: {len(prediction_dataset)}')
    return predictions