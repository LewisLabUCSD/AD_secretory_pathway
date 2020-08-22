# =============================================================================
import networkx as nx
from tqdm import tqdm

def read_network(network_file, delim = ','):
    # todo: add weighted network
    """
    Reads a network from an external file.

    * The edgelist must be provided as a tab-separated table. The
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


# =============================================================================
def read_gene_list(gene_file):
    """
    Reads a list genes from an external file.

    * The genes must be provided as a table. If the table has more
    than one column, they must be tab-separated. The first column will
    be used only.

    * Lines that start with '#' will be ignored
    """

    genes_set = set()
    for line in open(gene_file, 'r'):
        # lines starting with '#' will be ignored
        if line[0] == '#':
            continue
        # the first column in the line will be interpreted as a seed
        # gene:
        line_data = line.strip().split('\t')
        gene = line_data[0]
        genes_set.add(gene)

    print("\n> done reading genes:")
    print("> %s genes found in %s" % (len(genes_set), gene_file))

    return genes_set
