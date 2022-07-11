
from sklearn import metrics


def evaluate(self):
    """
    Compute, print and return:
    Overall metrics
    WC: mAP, AUC
    :return:
    """
    # todo review what's going on
    # Area Under the Receiver Operating Characteristic Curve (ROC AUC)
    auc = metrics.roc_auc_score(self.gt, self.predictions, average=None)

    # works in binary classification and multilabel indicator format
    average_precision = metrics.average_precision_score(self.gt, self.predictions, average=None)
    # average: None, the scores for each class are returned. Otherwise, determines type of averaging on the data:
    # 'micro': Calculate metrics globally by considering each element of the label indicator matrix as a label.
    # 'macro': Calculate metrics for each label, and find their unweighted mean. does not take label imbalance into account.
    # 'weighted': Calculate metrics for each label, and find their average, weighted by support
    # (ie, the number of true instances for each label).

    self.per_class_results = {'average_precision': average_precision,
                              'auc': auc}

    self.overall_results = {'map': np.mean(self.per_class_results['average_precision']),
                            'auc': np.mean(self.per_class_results['auc'])}