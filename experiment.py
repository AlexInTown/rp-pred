import os
import sys
import cPickle as cp
import pandas as pd
from sklearn import preprocessing, cross_validation



data_path = os.path.join(os.path.dirname(__file__), 'data')

class Experiment:
    def __init__(self):
        self.type_info = Experiment.load_feat_type_info()
        self.X = pd.load_csv(os.path.join(data_path,'train_x.csv'))
        self.Y = pd.load_csv(os.path.join(data_path, 'train_y.csv'))
        # x 和 y 值还得重新通过uid 对应起来
        self.X.sort(by='uid', inplace=True)
        self.Y.sort(by='uid', inplace=True)
        self.uid = self.X.uid.values
        self.X.drop(['uid'], axis = 1)
        assert self.uid == self.Y.uid.values, 
            "Error: uids in train_x.csv and train_y.csv not match! "
        self.Y.drop(['uid'], axis = 1)
        self.test = pd.load_csv(os.path.join(data_path, 'test_x.csv'))

        pass

    @classemthod
    def load_feat_type_info(cls, fname = os.path.join(data_path, 'features_type.csv')):
        f = open(fname, 'r')
        i = 0
        type_info = dict()
        for line in f:
            if i != 0:
                strs = line.rstrip(' \r\n').split(',')
                type_info[strs[0]]=strs[1]
            i += 1
        return type_info

    def process_category_feats(self):
        cate_cols = list()
        for col_name, col_type in self.type_info:
            if col_type != 'numeric':
                cate_cols.append(col_name)
        for col in cate_cols:
            cols = pd.get_dummies(self.X[col], prefix=col)
            self.X = pd.concat([self.X, cols], axis = 1)
        self.X.drop(cate_cols)
        return cate_cols


    def cross_validation(self, model, random_state=322435):
        kfold = cross_validation.KFold(self.X.size(), nfold=5, shuffle=True, random_state)
        scores = list()
        preds = list()
        for train_idx, test_idx in kfold:
            train_x = self.X[train_idx]
            train_y = self.Y[train_idx]
            test_x = self.X[test_idx]
            test_y = self.Y[test_idx]
            model.fit(train_x, train_y)
            pred = model.predict(test_x)
            score = metrics.roc_auc_score(test_y, pred) # ???
            preds.append(pred)
            scores.append(score)
        scores = np.asarray(scores)
        print scores.mean(), scores.std()
        return scores, preds






 
 