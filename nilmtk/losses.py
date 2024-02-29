from sklearn.metrics import mean_squared_error, mean_absolute_error, f1_score, r2_score, recall_score, precision_score, \
    matthews_corrcoef
import numpy as np

def mae(app_gt,app_pred):
    return mean_absolute_error(app_gt,app_pred)

def rmse(app_gt, app_pred):
    return mean_squared_error(app_gt,app_pred)**(.5)

def f1score(app_gt, app_pred):
    threshold = 10
    gt_temp = np.array(app_gt)
    gt_temp = np.where(gt_temp<threshold,0,1)
    pred_temp = np.array(app_pred)
    pred_temp = np.where(pred_temp<threshold,0,1)

    return f1_score(gt_temp, pred_temp)

def relative_error(app_gt,app_pred):
    constant = 1
    numerator = np.abs(app_gt - app_pred)
    denominator = constant + app_pred
    return np.mean(numerator/denominator)

def r2score(app_gt,app_pred):
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html#sklearn.metrics.r2_score
    return r2_score(app_gt, app_pred)

def nde(app_gt,app_pred):
    # Normalized Disaggregation Error (NDE)
    # Inspired by http://proceedings.mlr.press/v22/zico12/zico12.pdf
    numerator = np.sum((app_gt-app_pred)**2)
    denominator = np.sum(app_gt**2)

    return np.sqrt(numerator/denominator)

def nep(app_gt,app_pred):
    # Normalized Error in Assigned Power (NEP)
    # Inspired by https://www.springer.com/gp/book/9783030307813
    numerator = np.sum(np.abs(app_gt-app_pred))
    denominator = np.sum(app_gt)

    return numerator/denominator

# added from other files
def recall(app_gt, app_pred):
    threshold = 10
    gt_temp = np.array(app_gt)
    gt_temp = np.where(gt_temp<threshold,0,1)
    pred_temp = np.array(app_pred)
    pred_temp = np.where(pred_temp<threshold,0,1)

    return recall_score(gt_temp, pred_temp)

def precision(app_gt, app_pred):
    threshold = 10
    gt_temp = np.array(app_gt)
    gt_temp = np.where(gt_temp<threshold,0,1)
    pred_temp = np.array(app_pred)
    pred_temp = np.where(pred_temp<threshold,0,1)

    return precision_score(gt_temp, pred_temp)

def omae(app_gt, app_pred):
    threshold = 10
    gt_temp = np.array(app_gt)
    idx = gt_temp > threshold
    gt_temp = gt_temp[idx]
    pred_temp = np.array(app_pred)
    pred_temp = pred_temp[idx]

    return mae(gt_temp, pred_temp)

def MCC(app_gt, app_pred):
    threshold = 10
    gt_temp = np.array(app_gt)
    gt_temp = np.where(gt_temp<threshold,0,1)
    pred_temp = np.array(app_pred)
    pred_temp = np.where(pred_temp<threshold,0,1)

    return matthews_corrcoef(gt_temp, pred_temp)

def sae(app_gt, app_pred):
    '''
    compute the signal aggregate error
    sae = |\hat(r)-r|/r where r is the ground truth total energy;
    \hat(r) is the predicted total energy.
    '''
    sample_second = 6.0 # sample time is 6 seconds
    r = np.sum(app_gt * sample_second * 1.0 / 3600.0)
    rhat = np.sum(app_pred * sample_second * 1.0 / 3600.0)

    return np.abs(r - rhat) / np.abs(r)