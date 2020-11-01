import torch

class Dataset(torch.utils.data.Dataset):
  def __init__(self, identifier, trp_class, training_data):
        self.training_data = training_data
        self.trp_class = trp_class
        self.identifier = identifier

  def __len__(self):
        'return number of samples'
        return len(self.identifier)

  def __getitem__(self, index):
        'return sample of data and corresponding label'
        # Select sample
        id = self.identifier[index]

        # Load data and get label
        X = torch.load('data/{}/'.format(self.training_data) + id + '.pt')
        y = self.trp_class[id]

        return X, y
