class FeatureVectors(Dataset):
    """"
        This class concatenates the data from SF_GRU for pytorch dataset implementation
    """
    def __init__(self, data, type: STRING, ):
        super().__init__()    
        self.x_data = torch.from_numpy(np.concatenate((
            data[0][0], # local box     512
            data[0][1], # local contex  512
            data[0][2], # pose          36
            data[0][3], # bbox          4
            data[0][4]),# speed         1             
            axis = 2)).double()

        print('\n### ',type,'dataset number of pedestrians ###:',self.x_data.shape[0])

        self.y_data = torch.tensor(data[1]).double()

        self.n_samples = self.y_data.shape[0]
        self.type = type

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    
    def __len__(self):
        return self.n_samples
