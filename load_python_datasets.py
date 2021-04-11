def load_expr_mat(use_expr, project_root_dir='./', transformation=None, gene_symbol_col = 'geneSymbol'):
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
        expr_mat = pd.read_feather('%s/databases/AD_sc_counts.feather' % project_root_dir).set_index(
            gene_symbol_col)
    elif use_expr == 'MSBB_normExp':
        # NormalizedExpression
        expr_mat = pd.read_feather('%s/databases/AD_MSBB_scaleExp.feather' % project_root_dir).set_index(
            gene_symbol_col)
    else:
        if use_expr.endswith('.feather'):
            expr_mat = pd.read_feather('%s/%s' % (project_root_dir, use_expr)).set_index(
                gene_symbol_col)
        elif use_expr.endswith('.tsv.gz')  or use_expr.endswith('.tsv'):
            expr_mat = pd.read_csv('%s/%s' % (project_root_dir, use_expr), sep='\t').set_index(
                gene_symbol_col)
        elif use_expr.endswith('.csv.gz') or use_expr.endswith('.csv'):
            expr_mat = pd.read_csv('%s/%s' % (project_root_dir, use_expr)).set_index(
                gene_symbol_col)
        elif use_expr.startswith('http'):
            expr_mat = pd.read_csv(use_expr).set_index(gene_symbol_col)
        else:
            raise KeyError
        # raise KeyError

    if transformation is not None:
        ## TODO: non-zscore transformation
        if transformation == 'log':
            expr_mat = np.log1p(expr_mat)
        elif transformation == 'sigmoid':
            from scipy import stats, special
            expr_mat = special.expit(expr_mat.apply(stats.zscore))
        elif transformation == 'log_sigmoid':
            from scipy import stats, special
            expr_mat = special.expit(np.log1p(expr_mat).apply(stats.zscore))
        elif transformation == 'z_score_only':
            from scipy import stats
            expr_mat = expr_mat.apply(stats.zscore)
        else:
            raise KeyError

    return expr_mat
