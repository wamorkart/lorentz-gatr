import torch
import numpy as np
from torch_geometric.data import Data

EPS = 1e-5


class TaggingDataset(torch.utils.data.Dataset):
    """
    We use torch_geometric to handle point cloud of jet constituents more efficiently
    The torch_geometric dataloader concatenates jets along their constituent direction,
    effectively combining the constituent index with the batch index in a single dimension.
    An extra object batch.batch for each batch specifies to which jet the constituent belongs.
    We extend the constituent list by a global token that is used to embed extra global
    information and extract the classifier score.

    Structure of the elements in self.data_list
    x : torch.tensor of shape (num_elements, 4)
        List of 4-momenta of jet constituents
    scalars : empty placeholder
    label : torch.tensor of shape (1), dtype torch.int
        label of the jet (0=QCD, 1=top)
    is_global : torch.tensor of shape (num_elements), dtype torch.bool
        True for the global token (first element in constituent list), False otherwise
        We set is_global=None if no global token is used
    """

def __init__(self, 
        include_jet_data, 
        include_const_data, 
        global_jet_info, 
        const_jet_info
        ):
        super().__init__()
        self.include_jet_data = include_jet_data
        self.include_const_data = include_const_data
        self.global_jet_info = global_jet_info 
        self.const_jet_info = const_jet_info

        # Means and stds for standardization
        self.mean_const = [0.031, 0.0, -0.10,
                          -0.23,-0.10,0.27, 0.0,
                          0.0, 0.0,  0.0,  0.0,  0.0, 0.0]
        self.std_const = [0.35, 0.35,  0.178, 
                         1.2212526, 0.169,1.17,1.0,
                         1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        self.mean_jet =  [ 19.15986358 , 0.57154217 , 6.00354102, 11.730992]
        self.std_jet  = [9.18613789, 0.80465287 ,2.99805704 ,5.14910232]
        

    def load_data(self, filename, mode):
        raise NotImplementedError

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]


class TopTaggingDataset(TaggingDataset):
    def load_data(
        self,
        filename,
        mode,
        dtype=torch.float32,
    ):
        """
        Parameters
        ----------
        filename : str
            Path to file in npz format where the dataset in stored
        mode : {"train", "test", "val"}
            Purpose of the dataset
            Train, test and validation datasets are already seperated in the specified file
        """
        data = np.load(filename)
        kinematics = data[f"kinematics_{mode}"]
        labels = data[f"labels_{mode}"]

        kinematics = torch.tensor(kinematics, dtype=dtype)
        labels = torch.tensor(labels, dtype=torch.bool)

        # create list of torch_geometric.data.Data objects
        self.data_list = []
        for i in range(kinematics.shape[0]):
            # drop zero-padded components
            mask = (kinematics[i, ...].abs() > EPS).all(dim=-1)
            fourmomenta = kinematics[i, ...][mask]
            label = labels[i, ...]
            scalars = torch.zeros(
                fourmomenta.shape[0],
                0,
                dtype=dtype,
            )  # no scalar information
            jet_features = torch.zeros(
                fourmomenta.shape[0],
                0,
                dtype=dtype,
            )  # no jet features
            data = Data(x=fourmomenta, scalars=scalars, jet_features=jet_features, label=label)
            self.data_list.append(data)


class TopTaggingDataset_scalar(TaggingDataset):
    def load_data(
        self,
        filename,
        mode,
        dtype=torch.float32,
    ):
        """
        Parameters
        ----------
        filename : str
            Path to file in npz format where the dataset in stored
        mode : {"train", "test", "val"}
            Purpose of the dataset
            Train, test and validation datasets are already seperated in the specified file
        """
        data = np.load(filename)
        kinematics = data[f"kinematics_{mode}"]
        scalar_q2 = data[f"event_scalars_{mode}"]
        labels = data[f"labels_{mode}"]

        kinematics = torch.tensor(kinematics, dtype=dtype)
        scalar_q2 = torch.tensor(scalar_q2, dtype=dtype)
        labels = torch.tensor(labels, dtype=torch.bool)

        # create list of torch_geometric.data.Data objects
        self.data_list = []
        for i in range(kinematics.shape[0]):
            # drop zero-padded components
            mask = (kinematics[i, ...].abs() > EPS).all(dim=-1)
            fourmomenta = kinematics[i, ...][mask]
            scalars_mask = torch.zeros(
                fourmomenta.shape[0],
                0,
                dtype=dtype,
            )
            jet_features = torch.zeros(
                fourmomenta.shape[0],
                0,
                dtype=dtype,
            )
            if self.const_jet_info + self.global_jet_info > scalar_q2.shape[-1]:
                raise ValueError("Invalid scalar configuration")            
            if self.include_jet_data:
                jet_features = scalar_q2[i, 0, self.const_jet_info:]    
                jet_features = (jet_features - torch.tensor(self.mean_jet, dtype=jet_features.dtype)) / torch.tensor(self.std_jet, dtype=jet_features.dtype)            
            if self.include_const_data:
                scalars_mask = scalar_q2[i, :, :self.const_jet_info][mask]
                scalars_mask = (scalars_mask - torch.tensor(self.mean_const, dtype=scalars_mask.dtype)) / torch.tensor(self.std_const, dtype=scalars_mask.dtype)
            label = labels[i, ...]
            data = Data(x=fourmomenta, scalars=scalars_mask, jet_features=jet_features, label=label)
            self.data_list.append(data)


class QGTaggingDataset(TaggingDataset):
    def load_data(
        self,
        filename,
        mode,
        dtype=torch.float32,
    ):
        """
        Parameters
        ----------
        filename : str
            Path to file in npz format where the dataset in stored
        mode : {"train", "test", "val"}
            Purpose of the dataset
            Train, test and validation datasets are already seperated in the specified file
        """
        data = np.load(filename)
        kinematics = data[f"kinematics_{mode}"]
        pids = data[f"pid_{mode}"]
        labels = data[f"labels_{mode}"]

        kinematics = torch.tensor(kinematics, dtype=dtype)
        pids = torch.tensor(pids, dtype=dtype)
        labels = torch.tensor(labels, dtype=torch.bool)

        # create list of torch_geometric.data.Data objects
        self.data_list = []
        for i in range(kinematics.shape[0]):
            # drop zero-padded components
            mask = (kinematics[i, ...].abs() > EPS).all(dim=-1)
            fourmomenta = kinematics[i, ...][mask]
            scalars = pids[i, ...][mask]  # PID scalar information
            label = labels[i, ...]
            data = Data(x=fourmomenta, scalars=scalars, label=label)
            self.data_list.append(data)
