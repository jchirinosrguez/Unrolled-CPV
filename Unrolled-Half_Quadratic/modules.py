"""
module.
Class
-----------
MyDataset : manages loading data to be fed to the first layer of unrolled architecture.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import os


class MyDataset(torch.utils.data.Dataset):
    """"
    Loads the dataset.

    Attributes
    ----------
    Folder_Gr    (String)    : Groundtruth signals folder path
    Folder_De    (String)    : Degraded signals folder path
    file_names_Gr(List)      : List of groundtruth signals names
    file_names_De(List)      : List of degraded signals names
    file_list_Gr (List)      : List of groundtruth signal paths
    file_list_De (List)      : List of degraded signal paths
    """

    def __init__(self, folder,initial_x0, need_names):
        super(MyDataset, self).__init__()
        self.folder_Gr = os.path.join(folder, "Groundtruth")
        self.folder_De = os.path.join(folder, "Degraded")
        self.file_names_Gr = os.listdir(self.folder_Gr)
        self.file_names_De = os.listdir(self.folder_De)
        
        if self.file_names_Gr.count('.ipynb_checkpoints') !=0:
            self.file_names_Gr.remove('.ipynb_checkpoints')
        
        self.file_list_Gr = [os.path.join(self.folder_Gr, i) for i in self.file_names_Gr if not i.startswith('.')]
        self.file_list_De = [os.path.join(self.folder_De, i) for i in self.file_names_De if not i.startswith('.')]
        self.need_names = need_names
        self.initial_x0=initial_x0

    def __getitem__(self, index):
   
        X_true = np.load(self.file_list_Gr[index], allow_pickle=True)
        Degraded_path = (self.file_list_Gr[index].replace('Groundtruth', 'Degraded')).replace('Gr_', 'De_')
        X_degraded = np.load(Degraded_path, allow_pickle=True)
        if self.initial_x0 == "Wiener_initialization":
            Initial_path = (self.file_list_Gr[index].replace('Groundtruth', 'Initial')).replace('Gr_', 'In_')
            X_initial = np.load(Initial_path, allow_pickle=True)
            if self.need_names == 'no':
                return X_true, X_degraded,X_initial
            elif self.need_names == 'yes':
                name=os.path.splitext(self.file_names_Gr[index])[0]
                return name, X_true, X_degraded,X_initial
        else:

            if self.need_names == 'no':
                return X_true, X_degraded
            if self.need_names =="yes":
                name=os.path.splitext(self.file_names_Gr[index])[0]
                return name, X_true, X_degraded



    def __len__(self):

        return len(self.file_list_Gr)