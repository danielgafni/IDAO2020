import pandas as pd
import numpy as np
import torch
from utils import make_prediction, process_for_predict, targets


data = pd.read_csv('test.csv', parse_dates=['epoch'])
test_sat_id = np.unique(data['sat_id'].values)
sat_datas_test = process_for_predict(data)

submission = {}
submission['id'] = torch.tensor([]).type(torch.LongTensor)
for name in targets:
    submission[name] = torch.tensor([])

for data_test in sat_datas_test:
    sat_id = data_test['sat_id'].values[0]
    obs_ids = torch.from_numpy(data_test['id'].values)
    data_test = data_test.drop(['id', 'sat_id'], axis=1)
    prediction = make_prediction(data_test, sat_id)

    submission['id'] = torch.cat((submission['id'], obs_ids), axis=0)
    for i, name in enumerate(targets):
        submission[name] = torch.cat((submission[name],
                                      prediction[:, i]), axis=0)

submission = pd.DataFrame(submission)
submission.to_csv('submission_test.csv', index=False)