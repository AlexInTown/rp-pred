__author__ = 'tureoyz'

import copy
import sys
import random
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


def xgb_param_selection(exp, save_fold_res=0):
    print '============  xgb_param_selection  =============='
    param = {'bst:max_depth':10, 'bst:min_child_weight': 4, 'bst:subsample': 0.5, 'bst:colsample_bytree':0.8,  'bst:eta':0.05}
    other = {'silent':1, 'objective':'binary:logistic', 'nthread': 6, 'eval_metric': 'logloss', 'seed':0}
    model_param = copy.deepcopy(param)
    model_param.update(other)

    best_score = 0.0
    best_score_std = 0.0
    best_model_param = None
    for i in xrange(100):
        # random init params
        model_param['bst:max_depth'] = random.randint(7, 14)
        model_param['bst:min_child_weight'] = random.randint(1, 6)
        model_param['bst:subsample'] = random.uniform(0.6, 0.9)
        model_param['bst:colsample_bytree'] = random.uniform(0.4, 0.9)
        model_param['bst:eta'] = random.uniform(0.04, 0.13)

        xgb_model = XgboostModel(model_param)
        print '- training model {0} {1} '.format(i,  xgb_model.to_string())
        scores, preds = exp.cross_validation(xgb_model)
        avg_score, score_std = scores.mean(), scores.std()
        print avg_score, score_std
        if save_fold_res:
            model_name = xgb_model.to_string()
            pred_name = model_name + '.pkl'
            cp.dump((scores, preds), open(os.path.join('preds', pred_name), 'wb'), protocol=2)
        if avg_score > best_score:
            best_score = avg_score
            best_score_std =  score_std
            best_model_param = copy.deepcopy(model_param)
        pass
    print '- DONE ', best_score, best_score_std, best_model_param
    return best_score, best_score_std, best_model_param

def kmeans_param_selection(exp):
    exp.cross_validation()
    pass





if __name__=='__main__':
    print '------ loading experiment_l1 object  ----'
    exp_l1 = load_exp_l1()
    xgb_param_selection(exp_l1, save_fold_res=1)
    kmeans_param_selection(exp_l1)

