import math
import os
import sys
import warnings
import argparse

import numpy as np
import random

import pandas as pd
import torch
from lifelines import CoxPHFitter
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_ipcw, brier_score, cumulative_dynamic_auc

sys.path.append('./')

from api.coxkan import CoxKAN
from api.auton import preprocessing
from api.survset.data import SurvLoader
from api.baseline.dsm import DeepSurvivalMachines
from api.baseline.dcm import DeepCoxMixtures
from api.baseline.dcph import DeepCoxPH
from api.baseline.nsc import NeuralSurvivalCluster

if not sys.warnoptions: warnings.simplefilter("ignore")
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# matplotlib.style.use('default')

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, dest="seed")
parser.add_argument("--data", type=str, dest="data")
parser.add_argument("--model", type=str, dest="model")
parser.add_argument("--order", type=int, dest="order")
parser.add_argument("--lamb", type=float, dest="lamb")
args = parser.parse_args()

# random_state = 0
random_state = args.seed
random.seed(random_state)
np.random.seed(random_state)
torch.random.manual_seed(random_state)

device = 'cpu'

print('Model: {}\tData: {}\tSeed: {}\tLamb: {}\tOrder: {}'.format(
    args.model, args.data, args.seed, args.lamb, args.order))


def load_data(data=args.data):
    if data == 'linear' or data == 'non-linear':
        ranges, n_var, n_sample = [-1, 1], 2, 2000

        if data == 'linear':  # Linear Experiment
            f = lambda x: x[:, [0]] + x[:, [1]] * 2
        else:  # Non-Linear Experiment
            f = lambda x: math.log(5) * torch.exp(-(x[:, [0]] ** 2 + x[:, [1]] ** 2) / (2 * 0.5 ** 2))

        if len(np.array(ranges).shape) == 1:
            ranges = np.array(ranges * n_var).reshape(n_var, 2)
        else:
            ranges = np.array(ranges)

        train_input = torch.zeros(n_sample, n_var)
        for i in range(n_var):
            train_input[:, i] = torch.rand(n_sample, ) * (ranges[i, 1] - ranges[i, 0]) + ranges[i, 0]

        train_label = f(train_input)

        df = pd.concat([pd.DataFrame(train_input.numpy()), pd.DataFrame(train_label.numpy())], axis=1)
        df.columns = ['x1', 'x2', 'y']

        # randomly assign initial death time and individual death time
        df['initial_death'] = np.random.exponential(5, size=n_sample)
        df['ctime'] = df.initial_death / np.exp(df.y)

        df['event'] = 1  # generate events
        df.loc[df.ctime > np.quantile(df.ctime, 0.9), 'event'] = 0

        df['time'] = df['ctime']  # general final observation time
        df.loc[df.ctime > np.quantile(df.ctime, 0.9), 'time'] = np.quantile(df.ctime, 0.9)

        # return variables, time of observation, events of interest
        return df[['x1', 'x2']].values, df.time.values, df.event.values

    else:
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

        # return variables, time of observation, events of interest
        return features.values, df.time.values, df.event.values


if __name__ == '__main__':
    # ============================================= PREPARATION ========================================================
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

    # create file for logging metadata
    with open("./results/ablation/{}_{}_{}_{}_{}.txt".format(
            args.model, args.data, args.seed, args.lamb, args.order), "w") as file:
        file.write("model:\t{}\ndata:\t{}\nseed:\t{}\nseed:\t{}\nseed:\t{}".format(
            args.model, args.data, args.seed, args.lamb, args.order) + "\n\n")

    # =============================================== TRAINING =========================================================
    if args.model == 'coxkan':

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

        model = CoxKAN(width=[x_train.shape[1], 1], grid=5, k=args.order, seed=0)
        model.fit(dataset, opt=optimizer, steps=20, lr=lr, lamb=args.lamb, lamb_entropy=10.)

        logs = model.auto_symbolic(lib=['x', 'x^2', 'x^3', 'x^4', 'exp', 'log', 'sqrt', 'tanh', 'sin'])
        model.fit(dataset, opt=optimizer, lr=lr, steps=20)

        with open("./results/ablation/coxkan_symbol_{}_{}_{}_{}.txt".format(
                args.data, args.seed, args.lamb, args.order), "w") as file:
            file.write("model:\t{}\ndata:\t{}\nseed:\t{}".format(args.model, args.data, args.seed) + "\n\n")
            file.write("{}".format(logs) + "\n\n")
            file.write("{}".format(model.symbolic_formula(floating_digit=1)[0][0]) + "\n\n")
            file.write("{}".format(model.symbolic_formula(floating_digit=3)[0][0]) + "\n\n")
            file.write("{}".format(model.symbolic_formula(floating_digit=10)[0][0]))
            print(model.symbolic_formula(floating_digit=3)[0][0])

        out_risk = model.predict_risk(dataset['test_input'], times)
        out_survival = model.predict_survival(dataset['test_input'], times)

        # predict log-risk
        df_lrisk = (pd.concat([
            pd.DataFrame(dataset['test_input'].detach().numpy()),
            pd.DataFrame(model.forward(dataset['test_input']).detach().numpy())
        ], axis=1)).to_csv('./results/ablation/pred_{}_{}_{}_{}_{}.csv'.format(
            args.model, args.data, args.seed, args.lamb, args.order))

    else:

        # deep learning based survival models
        if args.model == 'dcph':
            model = DeepCoxPH(layers=[100, 100])
        elif args.model == 'dcm':
            model = DeepCoxMixtures(layers=[100, 100])
        elif args.model == 'dsm':
            model = DeepSurvivalMachines(layers=[100, 100])
        # Neural Survival Clustering (NSC) shows unsatisfying results in our experiments, thus excluded
        # elif args.model == 'nsc':
        #     model = NeuralSurvivalCluster(
        #         inputdim=x_train.shape[1], k=3, layers=[100, 100], act='ReLU6',
        #         layers_surv=[100, 100], representation=50, act_surv='Tanh')
        else:  # by default, use the DeepSurv or the DCPH model
            model = DeepCoxPH(layers=[100, 100])

        model.fit(x_train, t_train, e_train, iters=100, learning_rate=1e-4)

        out_risk = model.predict_risk(x_test, times)
        out_survival = model.predict_survival(x_test, times)

        if args.model == 'dcph':  # predict log-risk
            df_lrisk = (pd.concat([
                pd.DataFrame(x_test),
                pd.DataFrame(model.forward(x_test).detach().numpy())
            ], axis=1)).to_csv('./results/ablation/pred_{}_{}_{}_{}_{}.csv'.format(
                args.model, args.data, args.seed, args.lamb, args.order))

    # ============================================== EVALUATION ========================================================
    et_train = np.array([(e_train[i], t_train[i]) for i in range(len(e_train))], dtype=[('e', bool), ('t', float)])
    et_test = np.array([(e_test[i], t_test[i]) for i in range(len(e_test))], dtype=[('e', bool), ('t', float)])
    et_val = np.array([(e_val[i], t_val[i]) for i in range(len(e_val))], dtype=[('e', bool), ('t', float)])

    try:
        cis = []  # Concordance Index
        for i, _ in enumerate(times):
            cis.append(concordance_index_ipcw(et_train, et_test, out_risk[:, i], times[i])[0])

        # Brier Score
        brs = [brier_score(et_train, et_test, out_survival, times)[1]]

        roc_auc = []  # ROC AUC
        for i, _ in enumerate(times):
            roc_auc.append(cumulative_dynamic_auc(et_train, et_test, out_risk[:, i], times[i])[0])

        for horizon in enumerate(horizons):
            print("For {}-th quantile,".format(int(horizon[1] * 100)))
            print("C-Index:\t", cis[horizon[0]])
            print("Brier Score:\t", brs[0][horizon[0]])
            print("ROC AUC:\t", roc_auc[horizon[0]][0], "\n")

            # logging modeling results
            with open("./results/ablation/{}_{}_{}_{}_{}.txt".format(
                    args.model, args.data, args.seed, args.lamb, args.order), "a") as file:
                file.write(
                    "Quantile:\t{}".format(horizon[1]) +
                    "\nC-Index ({}):\t{}".format(horizon[1], cis[horizon[0]]) +
                    "\nBrier Score ({}):\t{}".format(horizon[1], brs[0][horizon[0]]) +
                    "\nROC AUC ({}):\t{}\n\n".format(horizon[1], roc_auc[horizon[0]][0]))

    except ValueError:
        os.remove("./results/ablation/{}_{}_{}_{}_{}.txt".format(
            args.model, args.data, args.seed, args.lamb, args.order))
        if args.model == 'coxkan':
            os.remove("./results/ablation/coxkan_symbol_{}_{}_{}_{}.txt".format(
                args.data, args.seed, args.lamb, args.order))
