import numpy as np
import pandas as pd
import torch
# from graph_assist import read_network, read_gene_list

import networkx as nx
from tqdm import tqdm


def read_network(network_file, delim=','):
    # todo: add weighted network
    """
    Reads a network from an external file.

    * The edgelist must be provided as a delim-separated table. The
    first two columns of the table will be interpreted as an
    interaction gene1 <==> gene2

    * Lines that start with '#' will be ignored
    """
    G = nx.Graph()
    if network_file.endswith('.gz'):
        import gzip
        handler = gzip.open(network_file, 'r')
    else:
        handler = open(network_file, 'r')
    next(handler) ## skep header

    for line in tqdm(handler):
        if type(line) is not str:
            line = line.decode('utf-8')
        # lines starting with '#' will be ignored
        if line[0] == '#':
            continue
        # The first two columns in the line will be interpreted as an
        # interaction gene1 <=> gene2
        line_data = line.strip().split(delim)
        node1 = line_data[0].strip('"')
        node2 = line_data[1].strip('"')
        G.add_edge(node1, node2)

    print("\n> done loading network:")
    print("> network contains %s nodes and %s links" % (G.number_of_nodes(),
                                                        G.number_of_edges()))

    return G


def RWR_torch(p0, G, expr, n_prop, a_prop, mask_on=[], mask_transparancy=0, add_loop=True,
              gradient: list = None, calculate_stationary=False,
              device=torch.device('cpu')):
    """

    :param add_loop: whether to add probability normalization loop for each node.
    :param mask_transparancy: 0-1 set to 0 to fully cloak the nodes not in mask_on; set to 1 to fully enable these
         nodes
    :type mask_on: list
    :param mask_on:
    :param p0:
    :param G: object of class ExprGraph
    :param expr:
    :param n_prop:
    :param a_prop:
    :param gradient: None to disable gradient calculation and return only p_k;
    set to True or a list of nodes to either return sum of gradients wrt to all nodes or specific sets of nodes
    :param device:
    :return:
    """

    if gradient is not None:
        requires_grad = True
    else:
        requires_grad = False
    # Adj = nx.to_numpy_array(G)
    # d_inv = np.array([1.0 / G.degree[n] for n in G.nodes])
    A = torch.tensor(G.Adj,
                     device=device,
                     dtype=torch.float
                     )
    if len(mask_on) > 0:
        if len(set(mask_on)) == len(mask_on):  # specify
            mask_vector_on = np.ones(len(G)) * mask_transparancy
            # mask_vector_on = np.zeros(len(G))
            mask_vector_on[[G.node_ind_dict[vt_node] for vt_node in mask_on]] = 1
        else:
            assert all([(x == 0) or (x == 1) for x in mask_on])
            mask_vector_on = mask_on

        # mask_off = [i for i, x in enumerate(mask_vector_on) if x == 0]
        m = torch.tensor(mask_vector_on,
                         device=device,
                         dtype=torch.float,
                         requires_grad=requires_grad)
        M = torch.diag(m)
        # M = torch.clamp(torch.diag(m), min=0, max=1)

        A_masked = torch.mm(torch.mm(M, A), M)  # A = MAM
        d = torch.sum(A_masked, dim=0)
        # d[mask_off] = 1.0

        D_inv = torch.diag(1.0 / d)
        W = torch.mm(A_masked, D_inv)

    else:
        D_inv = torch.diag(1.0 / torch.sum(A, dim=0))
        W = torch.mm(A, D_inv)
    expr = torch.tensor(expr, device=device, requires_grad=requires_grad, dtype=torch.float)
    p0 = torch.tensor(np.array(p0).reshape(len(G), 1),  # requires_grad=requires_grad,
                      device=device, dtype=torch.float)

    W_adj = torch.mm(torch.diag(expr), W)  # expr*W

    if add_loop:
        W_adj = W_adj + torch.diag(1 - torch.sum(W_adj, dim=0))

    p_k_last = p0
    for i in range(n_prop):
        p_k = (1 - a_prop) * torch.mm(W_adj, p_k_last) + a_prop * p0
        p_k_last = p_k

    return_dict = {'p_k': dict(zip(list(G.nodes),
                                   p_k.detach().cpu().numpy().reshape(-1)
                                   )
                               )}
    if calculate_stationary:
        stationary_dist_RWR3 = a_prop * torch.mm(torch.inverse(torch.diag(torch.ones(len(G),
                                                                                     device=device,
                                                                                     dtype=torch.float)) -
                                                               (1 - a_prop) * W_adj),
                                                 p0)
        adj_A = torch.mm(torch.diag(expr), A)
        stationary_dist = adj_A.sum(1) / torch.sum(adj_A, (0, 1))
        return_dict.update({
            'stationary_dist_RWR3': dict(zip(list(G.nodes),
                                             stationary_dist_RWR3.detach().cpu().numpy().reshape(-1))
                                         ),
            'stationary_dist': dict(zip(list(G.nodes),
                                        stationary_dist.detach().cpu().numpy().reshape(-1))
                                    )
        })

    if gradient is None:
        pass
    # elif gradient is True:
    #     torch.sum(p_k).backward()
    else:
        p_k.backward(
            torch.tensor(gradient, dtype=torch.float, device=device).reshape(len(G), 1))
        return_dict.update({'p_pk__p_expr': dict(zip(list(G.nodes),
                                                     expr.grad.detach().cpu().numpy().reshape(-1)
                                                     )
                                                 )
                            })
        if len(mask_on) > 0:
            return_dict.update({'p_pk__p_mask': dict(zip(list(G.nodes),
                                                         m.grad.detach().cpu().numpy().reshape(-1)
                                                         )
                                                     )
                                })

    return return_dict


class ExprGraph(nx.Graph):
    def __init__(self, G):
        super().__init__(G)
        self.node_ind_dict = {n: i for i, n in enumerate(G.nodes)}
        self.Adj = nx.to_numpy_array(self)
        self.D_inv = np.array([(1.0 / self.degree[n]) if self.degree[n] > 0 else 0 for n in self.nodes])

    def getExprVec(self, secP_secM_expr=None,
                   scale_expr_RWR3=True,
                   secM_expr_col='Proteins_iBAQ.rank.gene',
                   tissue='tissue.median'):
        """

        :param secP_secM_expr: pd.Dataframe or dictionary of gene/ protein expression; set to None to disable
        :param scale_expr_RWR3: whether to scale expression to [0,1]
        :param secM_expr_col:
        :param tissue:
        :return: expression vector
        """
        ## getting expression vector
        if secP_secM_expr is None:
            expr_dict = {x: 1 for x in range(len(self))}
            expr_filler = 1
            assert scale_expr_RWR3 is False

        elif type(secP_secM_expr) is dict:
            expr_dict = {k: v for k, v in secP_secM_expr.items() if k in self.nodes}
            expr_filler = np.median(list(expr_dict.values()))  # median imputation
        elif type(secP_secM_expr) is pd.DataFrame:
            context_expr = secP_secM_expr.loc[  ## imputation using only context nodes
                (secP_secM_expr['Tissue'] == tissue) & (secP_secM_expr['Gene name'].isin(self.nodes))]
            expr_dict = pd.Series(secP_secM_expr[secM_expr_col].values, index=secP_secM_expr['Gene name']).to_dict()
            expr_filler = np.median(context_expr[secM_expr_col].values)  # mean imputation
        else:
            raise TypeError('secP_secM_expr must be a dictionary or pd.Dataframe')

        expr_vector = np.array([expr_dict.get(n, expr_filler) for n in self.nodes])
        #
        # expr_vector[self.G.node_ind_dict[  ### source_expr_override=np.nan,
        #     self.secP]] = expr_filler  # force the expression of the source node to a given value in RWR3.
        if scale_expr_RWR3:
            expr_vector = (expr_vector - min(expr_vector)) / max((expr_vector - min(expr_vector)))  # normalization
        # else:
        #     assert all((0 <= expr_vector) & (expr_vector <= 1))

        return expr_vector


class RwrNode:

    def __init__(self, secp, G, secms, sec_resident=[], other_context_genes=[], remove_zero_degree=True, device='cuda'):
        G = G.subgraph(set([secp] + secms + sec_resident + other_context_genes))
        if remove_zero_degree:
            G = G.subgraph([n for (n, d) in G.degree() if d > 0])
        if secp not in G.nodes:
            raise Exception('secP isolated after context filtering')
        self.secP = secp
        self.G = ExprGraph(G)
        self.p0 = [1 if n == self.secP else 0 for n in self.G.nodes]  ## source vector
        self.secMs = list(set(secms).intersection(set(self.G.nodes)))
        self.sec_resident = list(set(sec_resident).intersection(set(self.G.nodes)))
        self.device = device

    def getRWR(self, n_prop, a_prop, secP_secM_expr=None, scale_expr_RWR3=True,
               secM_expr_col='Proteins_iBAQ.rank.gene',
               source_expr_override_mode=None, mask_on=[], mask_transparancy=0.0,
               vt_jacobian_nodes='secMs', n_sweep=None, add_loop=True, calculate_stationary=False):
        """

        :param add_loop: Add loop for probability normalization
        :param n_sweep: If set to an integer, return sweep analysis of p_k given mask_transparencies from 0-1
        :param mask_transparancy: 0-1 set to 0 to fully cloak the nodes not in mask_on; set to 1 to fully enable these
         nodes
        :type secP_secM_expr: dict, pd.Dataframe, None
        :type scale_expr_RWR3: bool
        :param scale_expr_RWR3: whether to scale the expression to 0-1
        :type vt_jacobian_nodes: basestring, list
        :param vt_jacobian_nodes: vector-Jacobian product
        set to 'secMs' to use only secMs, or to any list of proteins. Set to None to disable
        :return:
        :param n_prop:
        :param a_prop:
        :param secP_secM_expr: a pd.Dataframe or dictionary of expression values for each gene in the network.
        Set to None to disable (equivalent to using 1 for all genes
        :param source_expr_override_mode:  force the expression of the source node to a given value in RWR3
        set to None to disable, mean or median to set source expression to mean/ median of graph nodes expression
        """

        ## expression vector
        expr_vector = self.G.getExprVec(secP_secM_expr=secP_secM_expr, scale_expr_RWR3=scale_expr_RWR3,
                                        secM_expr_col=secM_expr_col,
                                        tissue='tissue.median')
        ### source_expr_override=np.nan,
        # .
        if source_expr_override_mode is not None:
            if source_expr_override_mode == 'mean':
                expr_vector[self.G.node_ind_dict[self.secP]] = np.mean(expr_vector)
            elif source_expr_override_mode == 'median':
                expr_vector[self.G.node_ind_dict[self.secP]] = np.median(expr_vector)

        if n_sweep is not None:

            ## sweep through expression
            p_k_list_exprsweep = []

            for expr_sweep_filler in np.linspace(0, 1, n_sweep):
                expr_sweep = expr_vector.copy()
                expr_sweep[[g not in mask_on for g in list(self.G.nodes)]] = expr_sweep_filler
                p_k_list_exprsweep.append(RWR_torch(p0=self.p0, G=self.G,
                                                    expr=expr_sweep, n_prop=n_prop, a_prop=a_prop,
                                                    gradient=None, device=torch.device(self.device)))

            ## sweep across mask transparencies
            p_k_list_mask_sweep = [RWR_torch(p0=self.p0, G=self.G,
                                             expr=expr_vector, n_prop=n_prop, a_prop=a_prop,
                                             gradient=None, device=torch.device(self.device), mask_on=mask_on,
                                             mask_transparancy=x) for x in tqdm(np.linspace(0, 1, n_sweep))]

            return {'expr_sweep': p_k_list_exprsweep,
                    'mask_sweep': p_k_list_mask_sweep}

        else:
            if vt_jacobian_nodes is None:
                return RWR_torch(p0=self.p0, G=self.G, expr=expr_vector,
                                 n_prop=n_prop, a_prop=a_prop,
                                 gradient=None, device=torch.device(self.device), mask_on=mask_on,
                                 mask_transparancy=None, add_loop=add_loop, calculate_stationary=calculate_stationary)

            elif vt_jacobian_nodes is 'secMs':
                vt_jacobian_nodes = self.secMs

            grad_one_hot = np.zeros(len(self.G))

            grad_one_hot[[self.G.node_ind_dict[vt_node] for vt_node in vt_jacobian_nodes]] = 1
            grad_one_hot /= sum(grad_one_hot)  ## normalization

            return RWR_torch(p0=self.p0, G=self.G, expr=expr_vector,
                             n_prop=n_prop, a_prop=a_prop,
                             gradient=grad_one_hot, device=torch.device(self.device), mask_on=mask_on,
                             mask_transparancy=mask_transparancy, add_loop=add_loop,
                             calculate_stationary=calculate_stationary)

    def sweep_analysis(self, secP_secM_expr, mask_of_node, n_sweep=100, add_loop=True):
        mask_on = [x for x in self.G.nodes if x not in ([mask_of_node] + [])]
        pkl = self.getRWR(n_prop=20, a_prop=.1,
                          secP_secM_expr=secP_secM_expr,
                          source_expr_override_mode='median',
                          vt_jacobian_nodes=None,
                          mask_on=mask_on,
                          mask_transparancy=None, n_sweep=n_sweep, add_loop=add_loop)
        secM_g_ids = [i for i, g in enumerate(list(self.G.nodes)) if g in self.secMs]

        return {
            sweep_type: [sweep_dict[x]['p_k'][secM_g_ids].sum() for x in range(n_sweep)] for sweep_type, sweep_dict in
            pkl.items()}

    def p_arwr(self, secP_secM_expr: dict,
               n_prop: int = 20, a_prop: float = .1,
               mask_on=[], mask_transparancy=.00001, summarization=False,
               vt_jacobian_nodes='secMs', calculate_stationary=False):

        """
        wrapper for expression guided random walk and gradient calculation

        :param a:
        :param secP_secM_expr:
        :param mask_on:
        :param mask_transparancy:
        :param summarization: set to 'secMs' to summarize p_k based on overall secM activities; or context_weighted_all to weigh
        all gene expression by network component score. Additionally, one can set summarization to context_weighted_secMs to limit
        summarization to only the secretory pathway components

        :return:
        """
        if summarization and calculate_stationary:
            # print('summary mode, stationary calculation unnecessary')
            calculate_stationary = False

        try:
            p_k_dict = self.getRWR(n_prop=n_prop, a_prop=a_prop, secP_secM_expr=secP_secM_expr,
                                   source_expr_override_mode='median', scale_expr_RWR3=True,
                                   vt_jacobian_nodes=vt_jacobian_nodes,
                                   mask_on=mask_on,
                                   mask_transparancy=mask_transparancy, add_loop=True,
                                   calculate_stationary=calculate_stationary)

            if summarization:
                p_k_dict = {'summarized': {'context_weighted_all':
                                               contextWeighted(a=self, p_k=p_k_dict['p_k'],
                                                               secP_secM_expr=secP_secM_expr),
                                           'context_weighted_secMs':
                                               contextWeighted(a=self,
                                                               p_k={secM: p_k_dict['p_k'].get(secM, 0)
                                                                    for
                                                                    secM in self.secMs},
                                                               secP_secM_expr=secP_secM_expr),
                                           'secM_avg': sum([p_k_dict['p_k'].get(secm, 0) for secm in self.secMs])
                                           }}

            return p_k_dict
        except RuntimeError as e:
            raise


def get_grad(secP, secP_secM_expr, mask_transparancy=.0001, other_context_genes=[], mask_off=[], add_loop=True):
    try:
        # other_context_genes = list(set(secPs) - {secP} - set(secMs) - set(sec_resident))
        # other_context_genes=[]
        a = RwrNode(
            secp=secP, G=G, secms=secMs, sec_resident=sec_resident, other_context_genes=other_context_genes,
        )

        mask_on = [x for x in a.G.nodes if x not in (mask_off + other_context_genes)]
        RWR = a.getRWR(n_prop=20, a_prop=.1,
                       secP_secM_expr=secP_secM_expr,
                       source_expr_override_mode='median',
                       vt_jacobian_nodes='secMs',
                       mask_on=mask_on,
                       mask_transparancy=mask_transparancy, add_loop=add_loop)
        return RWR
    except Exception as e:
        print(e)
        return None


def contextWeighted(a: RwrNode, p_k: dict, secP_secM_expr: dict):
    """
    generates secP-centrality weighted average of secM expression

    :param a:
    :param p_k:
    :param secP_secM_expr:
    :return:
    """
    imputed_expr_vector = a.G.getExprVec(secP_secM_expr, scale_expr_RWR3=True)
    expr_dict = dict(zip(a.G.nodes, imputed_expr_vector))
    imp_expr = np.median(list(expr_dict.values()))
    weighted_grad_avg = sum([grad * expr_dict.get(node, imp_expr) for node, grad in p_k.items()]) / sum(p_k.values())

    return weighted_grad_avg


def p_arwr(a: RwrNode, secP_secM_expr: dict,
           n_prop: int = 20, a_prop: float = .1,
           mask_on=[], mask_transparancy=.00001, summarization=False,
           vt_jacobian_nodes='secMs', calculate_stationary=False):
    """
    wrapper for expression guided random walk and gradient calculation

    :param a:
    :param secP_secM_expr:
    :param mask_on:
    :param mask_transparancy:
    :param summarization: set to True 'secMs' to summarize p_k based on overall secM activities; or context_weighted_all to weigh
    all gene expression by network component score. Additionally, one can set summarization to context_weighted_secMs to limit
    summarization to only the secretory pathway components

    :return:
    """
    try:
        p_k_dict = a.getRWR(n_prop=n_prop, a_prop=a_prop, secP_secM_expr=secP_secM_expr,
                            source_expr_override_mode='median', scale_expr_RWR3=True,
                            vt_jacobian_nodes=vt_jacobian_nodes,
                            mask_on=mask_on,
                            mask_transparancy=mask_transparancy, add_loop=True,
                            calculate_stationary=calculate_stationary)

        if summarization:
            p_k_dict = {'summarized': {'context_weighted_all':
                                           contextWeighted(a=a, p_k=p_k_dict['p_k'], secP_secM_expr=secP_secM_expr),
                                       'context_weighted_secMs':
                                           contextWeighted(a=a,
                                                           p_k={secM: p_k_dict['p_k'].get(secM, 0)
                                                                for
                                                                secM in a.secMs},
                                                           secP_secM_expr=secP_secM_expr),
                                       'secM_avg': sum([p_k_dict['p_k'].get(secm, 0) for secm in a.secMs])
                                       }}

        return p_k_dict
    except RuntimeError as e:
        raise


if __name__ == "__main__":
    from multiprocessing import Pool
    from functools import partial
    import socket
    import feather
    import pickle

    if socket.gethostname().startswith('chihchung-HP'):
        project_root_dir = '/home/chihchung/GoogleDrive/ppi'
        torch.set_num_threads(32)
        device = 'cuda'
    elif socket.gethostname().startswith('Chihchungs-MacBook-Pro'):
        project_root_dir = '/Volumes/GoogleDrive/My Drive/ppi'
        device = 'cpu'
    else:  # colab
        project_root_dir = '/root/GoogleDrive/My Drive/ppi'
        device = 'cuda'

    # G = read_network('%s/databases/networks/int.db.allfiltered.csv.gz' % project_root_dir)
    G = pickle.load(open('%s/databases/networks/int.db.PCNet.G.p' % project_root_dir, 'rb'))
    # ['ABCC8' in x for x in [list(G.neighbors('MAPT')), list(G.neighbors('APP'))]]

    # secP_secM_expr = feather.read_dataframe(
    #     '%s/python/data/deep_proteome_median/predicted.secP.HPA.feather' % project_root_dir)
    secMs = feather.read_dataframe(
        '%s/python/data/deep_proteome_median/secM.components.feather' % project_root_dir).iloc[:, 0].to_list()
    secPs = feather.read_dataframe(
        '%s/python/data/deep_proteome_median/secP.components.feather' % project_root_dir).iloc[:, 0].to_list()
    sec_resident = feather.read_dataframe(
        '%s/python/data/deep_proteome_median/all.secretory.resident.genes.feather' % project_root_dir).iloc[:,
                   0].to_list()

    APP_pathway_genes = pd.read_csv('%s/databases/2019_AD_GWAS/APP_pathway.csv' % project_root_dir)['Gene'].to_list()
    AD_risk_genes = pd.read_csv('%s/databases/2019_AD_GWAS/AD_risk_genes.csv' % project_root_dir)[
        'AD_risk_genes'].to_list()
    # secMs = list(set(secMs + APP_pathway_genes_df['Gene'].to_list()))
    # sec_resident = list(set(sec_resident + APP_pathway_genes_df['Gene'].to_list()))

    # candidateSecPs = ['APP', 'MAPT']
    candidateSecPs = ['APP',  # 'MAPT',
                      'ADAM10',  # 'ADAM10', 'ADAM17', 'ADAM19',
                      'BACE1',  # 'BACE2',
                      'PSEN1'  # , 'PSEN2', 'NCSTN', 'APH1A'
                      ]

    # G = G.subgraph(set(candidateSecPs + secMs + sec_resident))
    # expr_mat = pd.read_csv('%s/output/190908_AD/AD_sc_TPM.csv.gz' % project_root_dir)
    # expr_mat = pickle.load(open('%s/output/190908_AD/expr_mat_TPM.p' % project_root_dir, 'rb'))
    # expr_mat = pd.read_feather('%s/databases/2019_AD_singleCell/AD_sc_counts.feather' % project_root_dir).set_index('geneSymbol')
    expr_mat = pd.read_feather('%s/databases/2019_AD_MSBB/AD_MSBB_sigmoidExp.feather' % project_root_dir).set_index(
        'geneSymbol')
    # expr_mat=expr_mat
    # expr_arr = expr_mat.values

    patient_names = list(expr_mat.columns)
    patient_ids = range(expr_mat.shape[1])
    patient_name_dict = dict(zip(patient_ids, patient_names))
    allGeneSymbols = list(expr_mat.index)

    calculate_grad = False


    # expr_arr= stats.zscore(np.log(1+expr_arr))
    # expr_arr= expit(expr_arr)

    def rwr_run(secP):
        try:
            if calculate_grad:
                other_context_genes = list(
                    # set(G.nodes) - {secP} - set(secMs) - set(sec_resident)
                    set(APP_pathway_genes + AD_risk_genes) - {secP} - set(secMs) - set(sec_resident)
                    # set(secPs) - {secP} - set(secMs) - set(sec_resident)

                )  # list(set(candidateSecPs) - {secP} - set(secMs))
                a = RwrNode(secp=secP, G=G, secms=secMs, sec_resident=sec_resident,
                            other_context_genes=other_context_genes)
                mask_on = [x for x in a.G.nodes if x not in other_context_genes]
                vt_jacobian_nodes = 'secMs'
            else:
                other_context_genes = []
                a = RwrNode(secp=secP, G=G, secms=secMs, sec_resident=sec_resident,
                            other_context_genes=other_context_genes)
                mask_on = []
                vt_jacobian_nodes = None


        except Exception as e:
            # a = None
            # continue
            return None

        res_dict = {}
        for patient_id in patient_ids:
            res_dict.update({patient_name_dict[patient_id]:
                                 a.p_arwr(mask_on=mask_on,
                                          secP_secM_expr=dict(zip(allGeneSymbols, expr_mat.iloc[:, patient_id])),
                                          summarization=False,  # 'context_weighted_secM'
                                          vt_jacobian_nodes=vt_jacobian_nodes, calculate_stationary=True
                                          )})

        return res_dict


    with Pool(1) as p:
        res = p.map(rwr_run, candidateSecPs)

    res_frame = pd.concat([pd.DataFrame.from_dict({(secP, i, j): res_dict[i][j]
                                                   for i in res_dict.keys()
                                                   for j in res_dict[i].keys()},
                                                  orient='index') for secP, res_dict in zip(candidateSecPs, res) if
                           res_dict is not None])

    res_frame.to_csv('%s/output/190915_AD_MSBB/RWR_allSecPs_stationary.csv' % project_root_dir)
