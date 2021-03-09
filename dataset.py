import torch
import torch.utils.data
from torch_geometric.data import DataLoader
from torch_geometric.utils import to_networkx

from utils import compute_rwr


class DataLoaderGMN(DataLoader):
    def __init__(self, dataset, batch_size, shuffle, num_workers):
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        self.max_nodes = max(len(x.x) for x in dataset)
        self.num_features = dataset.num_features

    @staticmethod
    def _create_rwr(batch_list, max_nodes: int):
        out = torch.zeros(len(batch_list), max_nodes, max_nodes)
        for idx, g in enumerate(batch_list):
            rwr = torch.tensor(compute_rwr(to_networkx(g)))
            out[idx, :len(rwr), :len(rwr)] = rwr
        return out

    @staticmethod
    def _create_feats(batch_list, max_nodes: int, num_features: int):
        out = torch.zeros(len(batch_list), max_nodes, num_features)
        for idx, g in enumerate(batch_list):
            out[idx, :len(g.x)] = g.x
        return out

    def __iter__(self):
        for batch in super().__iter__():
            batch_list = batch.to_data_list()
            yield {
                # "adj": ...,  TODO: add this
                "num_nodes": torch.tensor([len(g.x) for g in batch_list]),
                "rwr": self._create_rwr(batch_list, self.max_nodes),
                "feats": self._create_feats(batch_list, self.max_nodes, self.num_features),
                "label": batch.y.squeeze(),
            }

# This class is partly borrowed from https://github.com/RexYing/diffpool
# class GraphSampler(torch.utils.data.Dataset):
#     def __init__(self, graphs, max_nodes, normalize=False):
#         self.adjs = []
#         self.features = []
#         self.rwrs = []
#         self.labels = []
#         self.max_nodes = max_nodes
#         self.feat_dim = len(graphs[0].nodes[list(graphs[0].nodes.keys())[0]]['feat'])
#
#         for graph in graphs:
#             num_nodes = graph.number_of_nodes()
#
#             adj = np.array(nx.to_numpy_matrix(graph))
#             if normalize:
#                 adj = normalize_adjacency(adj)
#             self.adjs.append(adj)
#
#             self.labels.append(graph.graph['label'])
#
#             feat = np.zeros((self.max_nodes, self.feat_dim), dtype=float)
#             feat[:num_nodes] = np.array(list(nx.get_node_attributes(graph, 'feat').values()))[:]
#             self.features.append(feat)
#
#             rwr = np.zeros((self.max_nodes, self.max_nodes), dtype=float)
#             src_array = np.array(list(nx.get_node_attributes(graph, 'rwr').values()))[:]
#             rwr[:src_array.shape[0], :src_array.shape[0]] = src_array
#             self.rwrs.append(-np.sort(-rwr, axis=1))
#
#         self.feat_dim = self.features[0].shape[1]
#
#     def __len__(self):
#         return len(self.adjs)
#
#     def __getitem__(self, idx):
#         adj = self.adjs[idx]
#         num_nodes = adj.shape[0]
#         adj_padded = np.zeros((self.max_nodes, self.max_nodes))
#         adj_padded[:num_nodes, :num_nodes] = adj[:]
#
#         return {'adj': adj_padded,
#                 'feats': self.features[idx].copy(),
#                 'rwr': self.rwrs[idx].copy(),
#                 'label': self.labels[idx],
#                 'num_nodes': num_nodes}
#
#
# class Dataset(object):
#     def __init__(self, name, max_nodes, num_folds, epsilon: float, random_seed: int):
#         self.name = name
#         self.train = None
#         self.val = None
#         self.batch_size = None
#         self.num_folds = num_folds
#         self.num_workers = 0
#         self.epsilon = epsilon
#         self.random_seed = random_seed
#
#         self.graphs = self.load(name, max_nodes, self.epsilon)
#         random.Random(self.random_seed).shuffle(self.graphs)
#         self.num_class = len(set([graph.graph['label'] for graph in self.graphs]))
#         self.max_nodes = max([graph.number_of_nodes() for graph in self.graphs])
#         self.feat_dim = len(self.graphs[0].nodes[list(self.graphs[0].nodes.keys())[0]]['feat'])
#
#     def process(self, batch_size, val_idx, normalize_adj):
#         self.batch_size = batch_size
#         val_size = len(self.graphs) // self.num_folds
#         train_graphs = self.graphs[:val_idx * val_size]
#
#         if val_idx < self.num_folds - 1:
#             train_graphs = train_graphs + self.graphs[(val_idx + 1) * val_size:]
#
#         val_graphs = self.graphs[val_idx * val_size: (val_idx + 1) * val_size]
#
#         dataset_sampler = GraphSampler(graphs=train_graphs, max_nodes=self.max_nodes, normalize=normalize_adj)
#
#         self.train = torch.utils.data.DataLoader(
#             dataset_sampler,
#             batch_size=self.batch_size,
#             shuffle=True,
#             pin_memory=True,
#             num_workers=self.num_workers)
#
#         dataset_sampler = GraphSampler(graphs=val_graphs, max_nodes=self.max_nodes, normalize=normalize_adj)
#
#         self.val = torch.utils.data.DataLoader(
#             dataset_sampler,
#             batch_size=self.batch_size,
#             shuffle=False,
#             pin_memory=True,
#             num_workers=self.num_workers)
#
#     def stat(self):
#         num_nodes, num_edges = zip(*[(g.number_of_nodes(), g.number_of_edges()) for g in self.graphs])
#         return dict(
#             nodes=dict(
#                 min=np.min(num_nodes),
#                 max=np.max(num_nodes),
#                 mean=np.mean(num_nodes),
#                 sd=np.std(num_nodes),
#             ),
#             edges=dict(
#                 min=np.min(num_edges),
#                 max=np.max(num_edges),
#                 mean=np.mean(num_edges),
#                 sd=np.std(num_edges),
#             ))
#
#     @staticmethod
#     def download(dataset):
#         basedir = os.path.dirname(os.path.abspath(__file__))
#         datadir = os.path.join(basedir, 'data', dataset)
#         if not os.path.exists(datadir):
#             print(f'Downloading {dataset} dataset ....')
#             os.makedirs(datadir)
#             url = 'https://ls11-www.cs.tu-dortmund.de/people/morris/graphkerneldatasets/{0}.zip'.format(dataset)
#             zipfile = os.path.basename(url)
#             os.system('wget {0}; unzip {1}'.format(url, zipfile))
#             os.system('mv {0}/* {1}'.format(dataset, datadir))
#             os.system('rm -r {0}'.format(dataset))
#             os.system('rm {0}'.format(zipfile))
#
#     # This function is partly borrowed from https://github.com/RexYing/diffpool
#     def load(self, dataset, max_nodes, epsilon: float):
#         self.download(dataset)
#         src = os.path.join(os.path.dirname(__file__), 'data')
#         prefix = os.path.join(src, dataset, dataset)
#
#         with open('{0}_graph_indicator.txt'.format(prefix), 'r') as f:
#             graph_node_dict = {idx + 1: int(line.strip('\n')) for idx, line in enumerate(f)}
#         max_nodes = min(Counter(graph_node_dict.values()).most_common(1)[0][1], max_nodes)
#
#         if os.path.exists('{0}_node_labels.txt'.format(prefix)):
#             with open('{0}_node_labels.txt'.format(prefix), 'r') as f:
#                 node_labels = [int(line.strip('\n')) - 1 for line in f]
#             num_unique_node_labels = max(node_labels) + 1
#         else:
#             print('No node labels')
#             node_labels = []
#
#         node_attrs = []
#         if os.path.exists('{0}_node_attributes.txt'.format(prefix)):
#             with open('{0}_node_attributes.txt'.format(prefix), 'r') as f:
#                 node_attrs = np.array(
#                     [np.array([float(attr) for attr in re.split("[,\s]+", line.strip("\s\n")) if attr])
#                      for line in f])
#             node_attrs -= np.mean(node_attrs, axis=0)
#             var = np.var(node_attrs, axis=0)
#             den = (var + epsilon) ** .5
#             node_attrs /= den
#         else:
#             print('No node attributes')
#
#         with open('{0}_graph_labels.txt'.format(prefix), 'r') as f:
#             graph_labels = [int(line.strip('\n')) for line in f]
#             unique_labels = set(graph_labels)
#         label_idx_dict = {val: idx for idx, val in enumerate(unique_labels)}
#         graph_labels = np.array([label_idx_dict[l] for l in graph_labels])
#
#         adj_list = {idx: [] for idx in range(1, len(graph_labels) + 1)}
#         index_graph = {idx: [] for idx in range(1, len(graph_labels) + 1)}
#         with open('{0}_A.txt'.format(prefix), 'r') as f:
#             for line in f:
#                 u, v = tuple(map(int, line.strip('\n').split(',')))
#                 adj_list[graph_node_dict[u]].append((u, v))
#                 index_graph[graph_node_dict[u]] += [u, v]
#
#         for k in index_graph.keys():
#             index_graph[k] = [u - 1 for u in set(index_graph[k])]
#
#         graphs = []
#         for idx in range(1, 1 + len(adj_list)):
#             graph = nx.from_edgelist(adj_list[idx])
#             if graph.number_of_nodes() > max_nodes:
#                 continue
#
#             graph.graph['label'] = graph_labels[idx - 1]
#             for u in graph.nodes():
#                 if len(node_labels) > 0:
#                     node_label_one_hot = [0] * num_unique_node_labels
#                     node_label = node_labels[u - 1]
#                     node_label_one_hot[node_label] = 1
#                     graph.nodes[u]['label'] = node_label_one_hot
#                 if len(node_attrs) > 0:
#                     graph.nodes[u]['feat'] = node_attrs[u - 1]
#             if len(node_attrs) > 0:
#                 graph.graph['feat_dim'] = node_attrs[0].shape[0]
#
#             mapping = {node: node_idx for node_idx, node in enumerate(graph.nodes())}
#             graphs.append(nx.relabel_nodes(graph, mapping))
#
#         if os.path.exists('{0}_rwrs.npy'.format(prefix)):
#             rwrs = np.load('{0}_rwrs.npy'.format(prefix), allow_pickle=True)
#         else:
#             print("No rwrs found. Computing them now...")
#             pool = Pool(cpu_count())
#             rwrs = pool.map(compute_rwr, graphs)
#             np.save('{0}_rwrs.npy'.format(prefix), rwrs)
#
#         if os.path.exists('{0}_rwrs.npy'.format(prefix)):
#             for graph_idx, graph in enumerate(graphs):
#                 assert graph.number_of_nodes() == len(rwrs[graph_idx])
#                 for node_idx, u in enumerate(graph.nodes()):
#                     graph.nodes[u]['rwr'] = rwrs[graph_idx][node_idx]
#
#         if 'feat_dim' in graphs[0].graph:
#             pass
#         elif 'label' in graphs[0].node[0]:
#             for graph in graphs:
#                 for u in graph.nodes():
#                     graph.node[u]['feat'] = np.array(graph.node[u]['label'])
#         else:
#             input_dim = 10
#             for graph in graphs:
#                 feat_dict = {idx: {'feat': np.ones(input_dim, dtype=float)} for idx in graph.nodes()}
#                 nx.set_node_attributes(graph, feat_dict)
#
#         return graphs
#
#
# if __name__ == '__main__':
#     datasets = ('ENZYMES', 'DD', 'REDDIT-MULTI-12K', 'COLLAB', 'PROTEINS_full')
#     epochs_ = 3
#     num_folds_ = 10
#     ds = Dataset(name=datasets[0], max_nodes=1000, num_folds=num_folds_, epsilon=1e-3, random_seed=0)
#
#     for fold_idx in range(num_folds_):
#         ds.process(batch_size=20, val_idx=fold_idx, normalize_adj=False)
#         for epoch in range(epochs_):
#             print('Epoch {0} --------------------'.format(epoch))
#             for batch_idx, batch_ in enumerate(ds.train):
#                 assert len(batch_['feats'][0][0]) == ds.feat_dim, '{0}-{1}'.format(len(batch_['feats'][0]), ds.feat_dim)
#                 print('Train', batch_idx, len(batch_['feats']), batch_['feats'].shape)
#             for batch_idx, batch_ in enumerate(ds.val):
#                 assert len(batch_['feats'][0][0]) == ds.feat_dim, '{0}-{1}'.format(len(batch_['feats'][0]), ds.feat_dim)
#                 print('Validation', batch_idx, len(batch_['feats']), batch_['feats'].shape)
