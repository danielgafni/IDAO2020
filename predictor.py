import pandas as pd
import numpy as np
import torch
from utils import make_prediction, process_for_predict, targets
from time import time as t

method = 'separated'

t_start = t()
data = pd.read_csv('test.csv', parse_dates=['epoch'])
test_sat_id = np.unique(data['sat_id'].values)
sat_datas_test = process_for_predict(data)

submission = {}
submission['id'] = torch.tensor([]).type(torch.LongTensor)
for name in targets:
    submission[name] = torch.tensor([])

model = torch.load(f'models//{0}//model.pt')
for sat_id in test_sat_id:
    data_test = sat_datas_test[sat_id]
    model.load_state_dict(torch.load(f'models//{sat_id}//state_dict.pt'))
    model.eval()
    obs_ids = torch.from_numpy(data_test['id'].values)
    data_test = data_test.drop(['id'], axis=1)
    print(f'Predicting for satellite {sat_id}')
    prediction = make_prediction(model=model, data=data_test, method=method)

    submission['id'] = torch.cat((submission['id'], obs_ids), axis=0)
    for i, name in enumerate(targets):
        submission[name] = torch.cat((submission[name],
                                      prediction[:, i]), axis=0)

submission = pd.DataFrame(submission)
submission.to_csv(f'submission-{method}.csv', index=False)
print(f'Total time: {t() - t_start}')
