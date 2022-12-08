import torch
import torch.nn.functional as F
import poptorch

from torch_geometric.data import Batch, Data
from torch_geometric.nn import to_fixed_size


class ShiftedSoftplus(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.shift = torch.log(torch.tensor(2.0)).item()

    def forward(self, x):
        u = torch.log1p(torch.exp(-x.abs()))
        v = torch.clamp_min(x, 0.)
        return u + v - self.shift

    @staticmethod
    def replace_activation(module: torch.nn.Module):
        import torch_geometric.nn.models.schnet as pyg_schnet
        for name, child in module.named_children():
            if isinstance(child, pyg_schnet.ShiftedSoftplus):
                setattr(module, name, ShiftedSoftplus())
            else:
                ShiftedSoftplus.replace_activation(child)


class TrainingModule(torch.nn.Module):
    """
    TrainingModule for SchNet.  Assumes that each mini-batch contains a single
    padding molecule at the end and uses this to calculate the mean squared
    error (MSE) for the real molcules in each mini-batch.
    """

    def __init__(self, module, batch_size, replace_softplus=True):
        super().__init__()
        if replace_softplus:
            ShiftedSoftplus.replace_activation(module)

        self.model = to_fixed_size(module=module, batch_size=batch_size)

    def forward(self, *args):
        args = [t.squeeze(0) for t in args]
        model_args, target = args[0:-1], args[-1]
        prediction = self.model(*model_args).view(-1)

        # slice off the padding molecule and calculate the mse loss
        prediction = prediction[0:-1]
        target = target[0:-1]
        loss = F.mse_loss(prediction, target)
        return prediction, loss


class KNNInteractionGraph(torch.nn.Module):

    def __init__(self, k: int, cutoff: float = 10.0):
        super().__init__()
        self.k = k
        self.cutoff = cutoff

    def forward(self, pos: torch.Tensor, batch: torch.Tensor):
        """
        k-nearest neighbors without dynamic tensor shapes

        :param pos (Tensor): Coordinates of each atom with shape
            [num_atoms, 3].
        :param batch (LongTensor): Batch indices assigning each atom to
                a separate molecule with shape [num_atoms]

        This method calculates the full num_atoms x num_atoms pairwise distance
        matrix. Masking is used to remove:
            * self-interaction (the diagonal elements)
            * cross-terms (atoms interacting with atoms in different molecules)
            * atoms that are beyond the cutoff distance

        Finally topk is used to find the k-nearest neighbors and construct the
        edge_index and edge_weight.
        """
        pdist = F.pairwise_distance(pos[:, None], pos, eps=0)
        rows = arange_like(batch.shape[0], batch).view(-1, 1)
        cols = rows.view(1, -1)
        diag = rows == cols
        cross = batch.view(-1, 1) != batch.view(1, -1)
        outer = pdist > self.cutoff
        mask = diag | cross | outer
        pdist = pdist.masked_fill(mask, self.cutoff)
        edge_weight, indices = torch.topk(-pdist, k=self.k)
        rows = rows.expand_as(indices)
        edge_index = torch.vstack([indices.flatten(), rows.flatten()])
        return edge_index, -edge_weight.flatten()


def arange_like(n: int, ref: torch.Tensor) -> torch.Tensor:
    return torch.arange(n, device=ref.device, dtype=ref.dtype)


def optimize_popart(options):
    """Apply a number of additional PopART options to optimize performace"""
    options._Popart.set('defaultBufferingDepth', 4)
    options._Popart.set("accumulateOuterFragmentSettings.schedule", 2)
    options._Popart.set(
        "replicatedCollectivesSettings.prepareScheduleForMergingCollectives",
        True)
    options._Popart.set(
        "replicatedCollectivesSettings.mergeAllReduceCollectives", True)
    return options


def prepare_data(data, target=4):
    """
    Prepares QM9 molecules for training SchNet for HOMO-LUMO gap prediction
    task.  Outputs a data object with attributes:

        z: the atomic number as a vector of integers with length [num_atoms]
        pos: the atomic position as a [num_atoms, 3] tensor of float32 values.
        y: the training target value. By default this will be the HOMO-LUMO gap
        energy in electronvolts (eV).
    """
    return Data(z=data.z, pos=data.pos, y=data.y[0, target].view(-1))


def padding_graph(num_nodes):
    """
    Create a molecule of non-interacting atoms defined as having atomic charge
    of zero to use for padding a mini-batch up to a known maximum size
    """
    assert num_nodes > 0
    return Data(z=torch.zeros(num_nodes, dtype=torch.long),
                pos=torch.zeros(num_nodes, 3),
                y=torch.zeros(1))


class CombinedBatchingCollator:
    """ Collator object that manages the combined batch size defined as:

        combined_batch_size = mini_batch_size * device_iterations
                             * replication_factor * gradient_accumulation

    This is intended to be used in combination with the poptorch.DataLoader
    """

    def __init__(self, mini_batch_size, keys, max_nodes_per_graph):
        """
        :param mini_batch_size (int): mini batch size used by the SchNet model
        :param keys: Keys to include from the batch in the
            output tuple specified as either a list or tuple of strings. The
            ordering of the keys is preserved in the tuple.
        """
        super().__init__()
        self.mini_batch_size = mini_batch_size
        self.keys = keys
        self.max_nodes_per_batch = max_nodes_per_graph * mini_batch_size

    def add_padding_graph(self, data_list):
        num_nodes = sum(d.num_nodes for d in data_list)
        num_pad_nodes = self.max_nodes_per_batch - num_nodes
        data_list.append(padding_graph(num_pad_nodes))
        return data_list

    def batch_to_tuple(self, data_list):
        batch = Batch.from_data_list(data_list)
        return tuple(getattr(batch, k) for k in self.keys)

    def __call__(self, batch):
        num_items = len(batch)
        assert num_items % self.mini_batch_size == 0, "Invalid batch size. " \
            f"Got {num_items} graphs and mini_batch_size={self.mini_batch_size}."

        num_mini_batches = num_items // self.mini_batch_size
        batches = [None] * num_mini_batches
        start = 0
        stride = self.mini_batch_size

        for i in range(num_mini_batches):
            slices = batch[start:start + stride]
            slices = self.add_padding_graph(slices)
            batches[i] = self.batch_to_tuple(slices)
            start += stride

        num_outputs = len(batches[0])
        outputs = [None] * num_outputs

        for i in range(num_outputs):
            outputs[i] = torch.stack(tuple(item[i] for item in batches))

        return tuple(outputs)


def create_dataloader(dataset,
                      ipu_opts=poptorch.Options(),
                      batch_size=2,
                      max_nodes_per_graph=32,
                      shuffle=False,
                      max_num_workers=32,
                      keys=("z", "pos", "batch", "y")):
    """
    Creates a data loader for graph datasets
    Applies the mini-batching method of concatenating multiple graphs into a 
    single graph with multiple disconnected subgraphs. See:
    https://pytorch-geometric.readthedocs.io/en/2.0.2/notes/batching.html
    
    Automatically adds a padding graph to fill up each mini-batch up to contain
    max_nodes_per_graph * (batch_size-1) nodes.
    """
    batch_size = batch_size - 1
    collater = CombinedBatchingCollator(batch_size, keys, max_nodes_per_graph)

    combined_batch_size = batch_size * ipu_opts.replication_factor * \
        ipu_opts.device_iterations * ipu_opts.Training.gradient_accumulation
    num_batches = len(dataset) // combined_batch_size
    num_workers = min(num_batches, max_num_workers)

    return poptorch.DataLoader(ipu_opts,
                               dataset=dataset,
                               batch_size=batch_size,
                               shuffle=shuffle,
                               num_workers=num_workers,
                               collate_fn=collater,
                               persistent_workers=num_workers > 0)
