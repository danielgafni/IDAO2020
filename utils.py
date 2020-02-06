import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
import plotly.graph_objs as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

features = ['epoch',
            'x_sim', 'y_sim', 'z_sim', 'Vx_sim', 'Vy_sim', 'Vz_sim',
            'ro_sim', 'theta_sim', 'fi_sim', 'dro/dt_sim', 'dtheta/dt_sim', 'dfi/dt_sim']
targets = ['x', 'y', 'z', 'Vx', 'Vy', 'Vz']


def smape(satellite_predicted_values, satellite_true_values):
    # Score function
    return torch.mean(torch.abs((satellite_predicted_values - satellite_true_values)
                          / (torch.abs(satellite_predicted_values) + torch.abs(satellite_true_values))))


class CustomDataset(Dataset):
    def __init__(self, data_train: pd.DataFrame, data_test: pd.DataFrame, seq_len=2):
        self.len = len(data_train) - (seq_len - 1)
        self.seq_len = seq_len
        self.x = torch.from_numpy(data_train.values).float()
        self.y = torch.from_numpy(data_test.values).float()

    def __len__(self):
        return self.len

    def __getitem__(self, item):
        return self.x[item:item + self.seq_len], self.y[item:item + self.seq_len]


def spherical_from_cartesian(data):
    data['ro_sim'] = np.sqrt(data['x_sim'] ** 2 + data['y_sim'] ** 2 + data['z_sim'] ** 2)
    data['theta_sim'] = np.arccos(data['x_sim'] / data['ro_sim'])
    data['fi_sim'] = np.arctan(data['z_sim'] / data['y_sim'])
    data['dro/dt_sim'] = (data['Vx_sim'] * data['x_sim'] + data['Vy_sim'] * data['y_sim'] + data['Vz_sim'] * data['z_sim']) / data['ro_sim']
    data['dtheta/dt_sim'] = (data['Vx_sim'] * data['y_sim'] - data['Vy_sim'] * data['x_sim']) / np.sqrt(data['x_sim'] ** 2 + data['y_sim'] ** 2)
    data['dfi/dt_sim'] = ((data['z_sim'] * (data['x_sim'] * data['Vx_sim'] + data['y_sim'] * data['Vy_sim']) - (data['x_sim'] ** 2 + data['y_sim'] ** 2) * data['Vz_sim'])
                          /(data['ro_sim'] ** 2 * np.sqrt(data['x_sim'] ** 2 + data['y_sim'] ** 2)))
    return data


def process(data):
    data = data.drop(['id'], axis=1)
    # Convert coordinates
    # data = spherical_from_cartesian(data)
    # Convert date and time to seconds
    data['epoch'] = data['epoch'].apply(lambda x: x.to_pydatetime().timestamp())
    data['epoch'] = (data['epoch'] - data['epoch'].min())
    # generate spherical coordinates features
    data = spherical_from_cartesian(data)
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