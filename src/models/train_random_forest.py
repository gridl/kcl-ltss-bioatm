import logging
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

import src.config.filepaths as fp


def main():

    trees_to_test = [32]
    bands = ['1', '4', '5', '6', '7', '10', '11', '12', '15']
    grads = ['1_grad', '4_grad', '5_grad', '6_grad',
             '7_grad', '10_grad', '11_grad', '12_grad', '15_grad']
    features = []
    for i in range(len(bands)):
        features.append(bands[i])
        features.append(grads[i])

    ds_path = os.path.join(fp.path_to_model_data_folder, 'rf_data.npy')
    ds = np.load(ds_path)

    X = ds[:, :-1]  # spectral data
    y = ds[:, -1]  # mask

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    for n_trees in trees_to_test:
        rf = RandomForestClassifier(n_estimators=n_trees, oob_score=True, n_jobs=3)
        rf = rf.fit(X_train, y_train)
        pickle.dump(rf,
                    open(os.path.join(fp.path_to_model_folder,
                                      'rf_model_{}_trees.pickle'.format(n_trees)), 'wb'))

        logfile_path = os.path.join(fp.path_to_model_folder, 'log_{}_trees.txt'.format(n_trees))

        print('OOB prediction of accuracy is: {oob}% \n'.format(oob=rf.oob_score_ * 100),
              file=open(logfile_path, 'a'))
        for c, imp in zip(features, rf.feature_importances_):
            print('Band {c} importance: {imp} \n'.format(c=c, imp=imp),
                  file=open(logfile_path, 'a'))

        df = pd.DataFrame()
        df['truth'] = y_test
        df['predict'] = rf.predict(X_test)

        # Cross-tabulate predictions
        print(pd.crosstab(df['truth'], df['predict'], margins=True),
              file=open(logfile_path, 'a'))
        print('\n', file=open(logfile_path, 'a'))

        # testing score
        score = metrics.f1_score(y_test, rf.predict(X_test))
        # training score
        score_train = metrics.f1_score(y_train, rf.predict(X_train))
        print(score, score_train, file=open(logfile_path, 'a'))
        print('\n', file=open(logfile_path, 'a'))


        pscore = metrics.accuracy_score(y_test, rf.predict(X_test))
        pscore_train = metrics.accuracy_score(y_train, rf.predict(X_train))

        print(pscore, pscore_train, file=open(logfile_path, 'a'))
        print('\n', file=open(logfile_path, 'a'))


if __name__ == "__main__":
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)
    main()