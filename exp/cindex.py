import os
import sys
import warnings
import argparse

# import matplotlib
import numpy as np
import random
import torch
from sksurv.metrics import concordance_index_censored, concordance_index_ipcw, brier_score, cumulative_dynamic_auc

sys.path.append('./')

from api.coxkan import CoxKAN
from api.auton import preprocessing
from api.survset.data import SurvLoader
from api.baseline.dsm import DeepSurvivalMachines
from api.baseline.dcm import DeepCoxMixtures
from api.baseline.dcph import DeepCoxPH

if not sys.warnoptions: warnings.simplefilter("ignore")
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# matplotlib.style.use('default')

parser = argparse.ArgumentParser()
# parser.add_argument("--seed", type=int, dest="seed")
parser.add_argument("--data", type=str, dest="data")
parser.add_argument("--model", type=str, dest="model")
args = parser.parse_args()

random_state = 0
# random_state = args.seed
random.seed(random_state)
np.random.seed(random_state)
torch.random.manual_seed(random_state)

device = 'cpu'


def load_data(data=args.data):
    loader = SurvLoader()

    # load dataset and its reference
    df, ref = loader.load_dataset(ds_name=data).values()
    df = df.sample(frac=1).reset_index(drop=True)

    # collect numerical and categorical features
    cat_feats, num_feats = [], []
    for i in df.columns:
        if i.split('_')[0] == 'num':
            num_feats.append(i)
        if i.split('_')[0] == 'fac':
            cat_feats.append(i)

    features = preprocessing.Preprocessor().fit_transform(
        df[num_feats + cat_feats], cat_feats=cat_feats, num_feats=num_feats)

    values, times, events = features.values, df.time.values, df.event.values

    return values, times, events


if __name__ == '__main__':
    x, t, e = load_data()

    horizons = [0.25, 0.5, 0.75]
    times = np.quantile(t[e == 1], horizons).tolist()

    n = len(x)

    tr_size = int(n * 0.70)
    vl_size = int(n * 0.10)
    te_size = int(n * 0.20)

    x_train, x_test, x_val = x[:tr_size], x[-te_size:], x[tr_size:tr_size + vl_size]
    t_train, t_test, t_val = t[:tr_size], t[-te_size:], t[tr_size:tr_size + vl_size]
    e_train, e_test, e_val = e[:tr_size], e[-te_size:], e[tr_size:tr_size + vl_size]

    if args.model == 'CoxKAN':
        dataset = {
            'train_input': torch.from_numpy(x_train).to(device),
            'val_input': torch.from_numpy(x_val).to(device),
            'test_input': torch.from_numpy(x_test).to(device),
            'train_time': torch.from_numpy(t_train).to(device),
            'val_time': torch.from_numpy(t_val).to(device),
            'test_time': torch.from_numpy(t_test).to(device),
            'train_event': torch.from_numpy(e_train).to(device),
            'val_event': torch.from_numpy(e_val).to(device),
            'test_event': torch.from_numpy(e_test).to(device)}

        optimizer = 'LBFGS'
        lr = 0.001

        model = CoxKAN(width=[x_train.shape[1], 1], grid=5, k=3, seed=0)
        model.fit(dataset, opt=optimizer, steps=20, lr=lr, lamb=0.01, lamb_entropy=10.)

        model.auto_symbolic(lib=['x', 'x^2', 'x^3', 'x^4', 'exp', 'log', 'sqrt', 'tanh'])
        model.fit(dataset, opt=optimizer, lr=lr, steps=20)
        print(model.symbolic_formula(floating_digit=3)[0][0])

        out_risk = model.predict_risk(dataset['test_input'], times)
        out_survival = model.predict_survival(dataset['test_input'], times)

    else:

        model = DeepCoxPH(layers=[100, 100])
        # model = DeepCoxMixtures(layers=[100, 100])

        # The fit method is called to train the model
        model.fit(x_train, t_train, e_train, iters=100, learning_rate=1e-4)
        # model = DeepSurvivalMachines(layers=[100, 100])

        out_risk = model.predict_risk(x_test, times)
        out_survival = model.predict_survival(x_test, times)

    et_train = np.array([(e_train[i], t_train[i]) for i in range(len(e_train))], dtype=[('e', bool), ('t', float)])
    et_test = np.array([(e_test[i], t_test[i]) for i in range(len(e_test))], dtype=[('e', bool), ('t', float)])
    et_val = np.array([(e_val[i], t_val[i]) for i in range(len(e_val))], dtype=[('e', bool), ('t', float)])

    cis = []  # Concordance Index
    for i, _ in enumerate(times):
        cis.append(concordance_index_ipcw(et_train, et_test, out_risk[:, i], times[i])[0])

    # Brier Score
    brs = [brier_score(et_train, et_test, out_survival, times)[1]]

    roc_auc = []  # ROC AUC
    for i, _ in enumerate(times):
        roc_auc.append(cumulative_dynamic_auc(et_train, et_test, out_risk[:, i], times[i])[0])

    for horizon in enumerate(horizons):
        print(f"For {horizon[1]} quantile,")
        print("TD Concordance Index:", cis[horizon[0]])
        print("Brier Score:", brs[0][horizon[0]])
        print("ROC AUC ", roc_auc[horizon[0]][0], "\n")
