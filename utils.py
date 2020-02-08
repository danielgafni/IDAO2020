import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
import plotly.graph_objs as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from models import LSTM
import os


features = ['dt',
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
    #  Separate different satellites
    data_grouped = data.groupby('sat_id')
    sat_datas_train = {}
    sat_datas_test = {}
    for sat_id, sat_data in data_grouped:
        sat_data.drop('sat_id', axis=1, inplace=True)
        sat_data.loc[:, 'epoch'] = sat_data['epoch'].apply(lambda x: x.timestamp())
        #  Convert epoch to time delta between observations
        sat_data.loc[:, 'dt'] = np.abs(sat_data.loc[:, 'epoch'].diff())
        sat_data.drop('epoch', axis=1, inplace=True)
        sat_data.iat[0, -1] = sat_data.iat[1, -1]
        sat_data.loc[:, 'dx_sim'] = sat_data.loc[:, 'Vx_sim'].mul(sat_data.loc[:, 'dt'])
        sat_data.loc[:, 'dy_sim'] = sat_data.loc[:, 'Vy_sim'].mul(sat_data.loc[:, 'dt'])
        sat_data.loc[:, 'dz_sim'] = sat_data.loc[:, 'Vz_sim'].mul(sat_data.loc[:, 'dt'])
        sat_data[targets + features] = sat_data[targets + features]
        sat_data_train, sat_data_test = train_test_split(sat_data.drop('id', axis=1), shuffle=False)
        scaler = StandardScaler(copy=False)
        scaler.fit(sat_data_train.loc[:, features])
        sat_data_train.loc[:, features] = scaler.transform(sat_data_train.loc[:, features])
        sat_data_test.loc[:, features] = scaler.transform(sat_data_test.loc[:, features])

        sat_datas_train[sat_id] = sat_data_train
        sat_datas_test[sat_id] = sat_data_test

        if not os.path.exists(f'models//{sat_id}'):
            os.makedirs(f'models//{sat_id}')
        torch.save(scaler, f'models//{sat_id}//StandardScaler')

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
    sat_datas = {}
    data_grouped = data.groupby('sat_id')
    for sat_data in data_grouped:
        sat_id = sat_data[1]['sat_id'].iloc[0]
        sat_datas[sat_id] = sat_data[1].drop(['sat_id'], axis=1)
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


def make_prediction(model, data, method='separated'):
    seq_len = 5
    prediction_dataset = PredictSequenceDataset(data, seq_len=seq_len)
    predictions = torch.zeros(len(data), 6)
    if method == 'separated':
        with torch.no_grad():
            for i in range(0, len(data) - seq_len, seq_len):
                seq_x = prediction_dataset[i].unsqueeze(0)
                # print(seq_x.shape)
                model.init_hidden_cell()
                predicted_seq = model(seq_x).squeeze(0)
                predictions[i:i+seq_len, :] = predicted_seq
            seq_x = prediction_dataset[len(prediction_dataset)-1]
            seq_x = seq_x.unsqueeze(0)
            model.init_hidden_cell()
            predicted_seq = model(seq_x).squeeze(0)
            predictions[-seq_len:, :] = predicted_seq

    if method == 'mean':
        count = torch.zeros(len(data), 6)
        with torch.no_grad():
            for i in range(0, len(prediction_dataset)):
                seq_x = prediction_dataset[i].unsqueeze(0)
                model.init_hidden_cell()
                predicted_seq = model(seq_x).squeeze(0)
                predictions[i:i+seq_len, :] += predicted_seq
                count[i:i+seq_len, :] += 1
            seq_x = prediction_dataset[len(prediction_dataset)-1]
            seq_x = seq_x.unsqueeze(0)
            model.init_hidden_cell()
            predicted_seq = model(seq_x).squeeze(0)
            predictions[-seq_len:, :] += predicted_seq
            count[-seq_len:, :] += 1
            print('----------------------')
            print(f'Predictions len: {predictions.shape[0]}')
            print(f'Real len: {len(data)}')

    if method == 'from_start':
        with torch.no_grad():
            for i in range(0, len(data) - seq_len):
                seq_x = prediction_dataset[i].unsqueeze(0)
                # print(seq_x.shape)
                model.init_hidden_cell()
                predicted_seq = model(seq_x).squeeze(0)
                predictions[i:i+seq_len, :] = predicted_seq

    return predictions
