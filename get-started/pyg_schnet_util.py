import torch
import torch.nn.functional as F
import poptorch

from torch_geometric.data import Batch, Data
from torch_geometric.nn import to_fixed_size
from torch_geometric.transforms import BaseTransform, Compose


class TrainingModule(torch.nn.Module):

    def __init__(self, module, batch_size) -> None:
        super().__init__()
        self.model = to_fixed_size(module=module, batch_size=batch_size)

    def forward(self, *args):
        args = [t.squeeze(0) for t in args]
        model_args, target = args[0:-1], args[-1]
        prediction = self.model(*model_args).view(-1)
        loss = self.mse_loss(prediction, target)
        return prediction, loss

    @staticmethod
    def mse_loss(input, target):
        """
        Calculates the mean squared error

        This loss assumes that zeros are used as padding on the target so that
        the count can be derived from the number of non-zero elements.
        """
        loss = F.mse_loss(input, target, reduction="sum")
        N = (target != 0.0).to(loss.dtype).sum()
        loss = loss / N
        return poptorch.identity_loss(loss, reduction="none")


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


class PadMolecule(BaseTransform):
    """
    Data transform that applies padding to enforce consistent tensor shapes.

    Padding atoms are defined as have atomic charge of zero and are placed at a 
    distance of 10 * cutoff ^ 2 to ensure there are no edges created between 
    """

    def __init__(self, max_num_atoms: int, cutoff: float):
        """
        :param max_num_atoms (int): The maximum number of atoms to pad
            the atomic numbers and position tensors to.
        :param cutoff (float): The cutoff in Angstroms used in the SchNet model.
        """
        super().__init__()
        self.max_num_atoms = max_num_atoms
        self.cutoff = cutoff

    def validate(self, data):
        """
        Validates that the input molecule does not exceed the constraint that
        the number of atoms must be <= max_num_atoms
        """
        num_atoms = data.z.numel()

        assert num_atoms <= self.max_num_atoms, \
            f"Too many atoms. Molecule has {num_atoms} atoms "\
            f"and max_num_atoms is {self.max_num_atoms}."

        return num_atoms

    def __call__(self, data):
        num_atoms = self.validate(data)
        num_fake_atoms = self.max_num_atoms - num_atoms
        data.z = F.pad(data.z, (0, num_fake_atoms), 'constant', 0)
        data.pos = F.pad(data.pos, (0, 0, 0, num_fake_atoms), 'constant',
                         10. * self.cutoff**2)
        data.num_nodes = self.max_num_atoms
        return data


class PrepareData(BaseTransform):
    """
    Data transform for preparing each data instance by:
        * extracting a given target label from the PyG QM9 dataset
        * slicing the data objet to only contain the provided properties.

    The QM9 dataset consists of a total of 19 regression targets. This transform
    indexes the regression targets stored in data.y to only include the selected
    target.

    Expected input:
        data.y is a vector with shape [1, 19]

    Transformed output:
        data.y is as scalar with shape torch.Size([])
    """

    def __init__(self, target, keys):
        self.target = target
        self.keys = keys

    def validate(self, data):
        assert hasattr(data, "y") \
          and isinstance(data.y, torch.Tensor) \
          and data.y.shape == (1, 19),\
          "Invalid data input. Expected data.y == Tensor with shape [1, 19]"

    def __call__(self, data):
        self.validate(data)
        data.y = data.y[0, self.target]

        values = [getattr(data, k) for k in self.keys]
        kwargs = dict([*zip(self.keys, values)])
        return Data(**kwargs)


class CombinedBatchingCollator:
    """ Collator object that manages the combined batch size defined as:

        combined_batch_size = mini_batch_size * device_iterations
                             * replication_factor * gradient_accumulation

    This is intended to be used in combination with the poptorch.DataLoader
    """

    def __init__(self, mini_batch_size, keys):
        """
        :param mini_batch_size (int): mini batch size used by the SchNet model
        :param keys: Keys to include from the batch in the
            output tuple specified as either a list or tuple of strings. The
            ordering of the keys is preserved in the tuple.
        """
        super().__init__()
        self.mini_batch_size = mini_batch_size
        self.keys = keys

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
            batches[i] = self.batch_to_tuple(slices)
            start += stride

        num_outputs = len(batches[0])
        outputs = [None] * num_outputs

        for i in range(num_outputs):
            outputs[i] = torch.stack(tuple(item[i] for item in batches))

        return tuple(outputs)


def create_transform(cutoff: float, max_num_atoms):
    """
    Creates a sequence of transforms defining a data pre-processing pipeline

    :param cutoff (float): Cutoff distance for interatomic interactions in
        Angstroms (default: 6.0).
    :param max_num_atoms (int): The maximum number of atoms used by the
        PadMolecule transform (default: 32).

    :returns: A composite transform
    """
    # The HOMO-LUMO gap is target 4 in QM9
    target = 4
    keys = ("z", "pos", "y", "num_nodes")

    return Compose([
        PadMolecule(cutoff=cutoff, max_num_atoms=max_num_atoms),
        PrepareData(target=target, keys=keys)
    ])


def create_dataloader(dataset,
                      ipu_opts,
                      batch_size=1,
                      shuffle=False,
                      num_workers=0):
    """
    Creates a data loader for graph datasets
    Applies the mini-batching method of concatenating multiple graphs into a 
    single graph with multiple disconnected subgraphs. See:
    https://pytorch-geometric.readthedocs.io/en/2.0.2/notes/batching.html
    """
    if ipu_opts is None:
        ipu_opts = poptorch.Options()

    keys = ("z", "pos", "batch", "y")
    collater = CombinedBatchingCollator(batch_size, keys)

    return poptorch.DataLoader(ipu_opts,
                               dataset=dataset,
                               batch_size=batch_size,
                               shuffle=shuffle,
                               num_workers=num_workers,
                               collate_fn=collater)
