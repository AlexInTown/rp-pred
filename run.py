__author__ = 'tureoyz'

import copy
import sys
import os
from experiment import *
from model_wrappers import *
import cPickle as cp


def load_exp_l1(exp1_path=None):
    try:
        exp1_path = os.path.join(os.path.dirname(__file__), 'data/exp1.pkl')
        exp_l1 = cp.load(open(exp1_path, 'rb'))
    except Exception:
        print 'Can not load experiment_l1 object from path %s' % exp1_path
        exp_l1 = ExperimentL1()
        print 'Create a new one and save to file %s' % exp1_path
        cp.dump(exp_l1, open(exp1_path, 'wb'), protocol=2)
    return exp_l1


def xgb_param_selection(exp):
    print '============  xgb_param_selection  =============='
    param = {'bst:max_depth':10, 'bst:min_child_weight': 4, 'bst:subsample': 0.5, 'bst:colsample_bytree':0.8,  'bst:eta':0.05}
    other = {'silent':1, 'objective':'binary:logistic', 'nthread': 6, 'eval_metric': 'logloss', 'seed':0}
    model_param = copy.deepcopy(param)
    model_param.update(other)
    xgb_model = XgboostModel(model_param)
    print '- training model %s '% xgb_model.to_string()
    loss_mean, loss_std = exp.cross_validation(xgb_model)
    pass

def kmeans_param_selection(exp):
    exp.cross_validation()
    pass





if __name__=='__main__':
    print '------ loading experiment_l1 object  ----'
    exp_l1 = load_exp_l1()
    xgb_param_selection(exp_l1)
    kmeans_param_selection(exp_l1)