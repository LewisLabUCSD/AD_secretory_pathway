def load_expr_mat(use_expr, project_root_dir='~/', transformation=None):
    """
    :param use_expr: 'SC_tpm', 'SC_counts', 'ROSMAP_fpkm', 'MSBB_normExp'
    :param project_root_dir:
    :param transformation: None, 'log', 'sigmoid'
    :return:
    """
    import pandas as pd
    import pickle
    import numpy as np

    if use_expr == 'SC_tpm':
        expr_mat = pickle.load(open('%s/output/190908_AD/expr_mat_TPM.p' % project_root_dir, 'rb'))
    elif use_expr == 'SC_counts':
        expr_mat = pd.read_feather('%s/databases/2019_AD_singleCell/AD_sc_counts.feather' % project_root_dir).set_index(
            'geneSymbol')
    elif use_expr == 'ROSMAP_fpkm':
        expr_mat = pd.read_feather('%s/databases/2019_AD_ROSMAP/AD_ROSMAP_FPKM.feather' % project_root_dir).set_index(
            'geneSymbol')
    elif use_expr == 'MSBB_normExp':
        # NormalizedExpression
        expr_mat = pd.read_feather('%s/databases/2019_AD_MSBB/AD_MSBB_scaleExp.feather' % project_root_dir).set_index(
            'geneSymbol')
    else:
        expr_mat = pd.read_feather('%s/%s' % (project_root_dir, use_expr)).set_index(
            'geneSymbol')
        # raise KeyError

    if transformation is not None:
        if transformation == 'log':
            expr_mat = np.log(1 + expr_mat)
        elif transformation == 'sigmoid':
            from scipy import stats, special
            expr_mat = special.expit(expr_mat.apply(stats.zscore))
        elif transformation == 'log_sigmoid':
            from scipy import stats, special
            expr_mat = special.expit(np.log(1 + expr_mat).apply(stats.zscore))
        else:
            raise KeyError

    return expr_mat
