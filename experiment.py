
# -*- coding: UTF-8 -*-
import os
import sys
import cPickle as cp
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn import preprocessing, cross_validation, metrics



data_path = os.path.join(os.path.dirname(__file__), 'data')

class ExperimentL1:
    def __init__(self, random_state=322435):
        self.type_info = ExperimentL1.load_feat_type_info()
        self.X = pd.read_csv(os.path.join(data_path,'train_x.csv'))
        self.Y = pd.read_csv(os.path.join(data_path, 'train_y.csv'))
        # x 和 y 值还得重新通过uid 对应起来
        self.X.sort(columns='uid', inplace=True)
        self.Y.sort(columns='uid', inplace=True)
        self.uid = self.X.uid.values
        self.X = self.X.drop(['uid'], axis = 1)
        assert np.array_equal(self.uid, self.Y.uid.values), \
            "Error: uids in train_x.csv and train_y.csv not match! "
        self.Y = self.Y.drop(['uid'], axis = 1)
        self.test = pd.read_csv(os.path.join(data_path, 'test_x.csv'))
        self.process_category_feats()
        self.random_state = random_state
        pass

    @classmethod
    def load_feat_type_info(cls, fname = os.path.join(data_path, 'features_type.csv')):
        f = open(fname, 'r')
        i = 0
        type_info = dict()
        for line in f:
            if i != 0:
                strs = line.rstrip(' \r\n').split(',')
                type_info[strs[0].strip('"')]=strs[1].strip('"')
            i += 1
        return type_info

    def process_category_feats(self):
        cate_cols = list()
        for col_name, col_type in self.type_info.iteritems():
            if col_type != 'numeric':
                cate_cols.append(col_name)
        for col in cate_cols:
            cols = pd.get_dummies(self.X[col], prefix=col)
            self.X = pd.concat([self.X, cols], axis=1)
            cols = pd.get_dummies(self.test[col], prefix=col)
            self.test = pd.concat([self.test, cols], axis=1)
        self.X = self.X.drop(cate_cols, axis=1)
        self.test = self.test.drop(cate_cols, axis=1)
        return cate_cols


    def cross_validation(self, model):
        kfold = cross_validation.KFold(self.X.shape[0], n_folds=5, shuffle=True, random_state=self.random_state)
        scores = list()
        preds = list()
        i = 0
        for train_idx, test_idx in kfold:
            print 'fold ', i
            train_x = self.X.iloc[train_idx]
            train_y = self.Y.iloc[train_idx]
            test_x = self.X.iloc[test_idx]
            test_y = self.Y.iloc[test_idx]
            model.fit(train_x, train_y)
            pred = model.predict(test_x)
            score = metrics.roc_auc_score(test_y, pred) # ???
            preds.append(pred)
            scores.append(score)
            i += 1
        scores = np.asarray(scores)
        print scores.mean(), scores.std()
        return scores, preds


class ExperimentL2:
    def __init__(self):
        # TODO load model + result pkl, use the model to get output attributes
        # traverse the same pkl file pattern
        # parse the meta feature
        # make dataset
        # cross validation to select out parameters
        self.random_stat = 2788863
        pass

    def cross_validation(self, model):
        kfold = cross_validation.KFold(self.X.shape[0], n_folds=5, shuffle=True, random_state=self.random_state)
        scores = list()
        preds = list()
        for train_idx, test_idx in kfold:
            train_x = self.X.iloc[train_idx]
            train_y = self.Y.iloc[train_idx]
            test_x = self.X.iloc[test_idx]
            test_y = self.Y.iloc[test_idx]
            model.fit(train_x, train_y)
            pred = model.predict(test_x)
            score = metrics.roc_auc_score(test_y, pred) # ???
            preds.append(pred)
            scores.append(score)
        scores = np.asarray(scores)
        print scores.mean(), scores.std()
        return scores, preds


