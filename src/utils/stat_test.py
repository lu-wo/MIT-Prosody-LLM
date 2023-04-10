import numpy as np
from scipy import stats


def f_oneway(res_model1, res_model2):
    """Perform one-way ANOVA.

    Args:
        res_model1 (list): List of residuals from model 1.
        res_model2 (list): List of residuals from model 2.

    Returns:
        float: F-statistic.
        float: p-value.
    """
    F, p = stats.f_oneway(res_model1, res_model2)
    return F, p
