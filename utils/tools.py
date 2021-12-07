import logging, yaml, argparse
import torch
import torch.nn as nn

def get_log(log_file):
    logger = logging.getLogger(log_file)
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger


def get_config(config_path="config.yml"):
    with open(config_path, "r") as setting:
        config = yaml.load(setting)
    return config


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/base.yaml', type=str, help='path to the config file')
    parser.add_argument('--rank', type=int, default=0, help='rank')
    parser.add_argument('--world_size', type=int, default=1, help='world size')
    parser.add_argument('--dist_backend', default='nccl', type=str, help='distributed backend')
    parser.add_argument('--dist_url', default='tcp://localhost:23456', type=str,
                        help='url used to set up distributed training')
    parser.add_argument("--vh_weight", type=float, default=0.5, help="")
    parser.add_argument("--pos_weight", type=float, default=0.1, help="")
    parser.add_argument("--ans_weight", type=float, default=0.1, help="")

    cfg = parser.parse_args()
    args = get_config(cfg.config)
    update_values(args, vars(cfg))
    cfg.learning_rate = float(cfg.learning_rate)
    return cfg


def update_values(dict_from, dict_to):
    for key, value in dict_from.items():
        if isinstance(value, dict):
            update_values(dict_from[key], dict_to[key])
        elif value is not None:
            dict_to[key] = dict_from[key]


def adj_dense2edge_sparse(adj: torch.Tensor):
    """

    Parameters
    ----------
    adj: torch.Tensor, shape=[Batch, Node_Number, Node_Number]

    Returns
    -------

    """
    assert len(adj.shape) == 3
    assert adj.shape[1] == adj.shape[2]
    device = adj.device
    max_number = adj.shape[1] * adj.shape[2]
    node_number = adj.shape[1]
    node2edge = torch.zeros(adj.shape[0], max_number, node_number)
    edge2node = torch.zeros(adj.shape[0], node_number, max_number)
    batch_max_edge_number = -1
    for i in range(adj.shape[0]):
        src, tgt = adj[i, :, :].nonzero(as_tuple=True)
        num_edges = src.shape[0]
        batch_max_edge_number = max(batch_max_edge_number, num_edges)
        edge_index = torch.Tensor(range(0, num_edges))
        src_indice = torch.zeros(num_edges, node_number)
        src_indice[edge_index.long(), src] = 1

        tgt_indice = torch.zeros(num_edges, node_number)
        tgt_indice[edge_index.long(), tgt] = 1

        node2edge[i, :num_edges, :] = tgt_indice
        edge2node[i, :, :num_edges] = src_indice.transpose(0, 1)
    node2edge = node2edge[:, :batch_max_edge_number, :]
    edge2node = edge2node[:, :, :batch_max_edge_number]
    node2edge = node2edge.to(device)
    edge2node = edge2node.to(device)

    return node2edge, edge2node


if __name__ == "__main__":
    a = torch.zeros(1, 3, 3)
    a[0, 0, 1] = 1
    a[0, 1, 2] = 1
    adj_dense2edge_sparse(a)
