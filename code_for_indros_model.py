# code for indro's model 

# import library
import torch
from torch import nn


# functions to load model
# create submodels
class get_submodel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(3, 16)
        torch.nn.init.xavier_uniform_(self.linear1.weight)
        self.linear2 = nn.Linear(16, 64)
        torch.nn.init.xavier_uniform_(self.linear2.weight)
        self.linear3 = nn.Linear(64, 256)
        torch.nn.init.xavier_uniform_(self.linear3.weight)
        self.linear4 = nn.Linear(256, 256)
        torch.nn.init.xavier_uniform_(self.linear4.weight)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(p=0.1)
        self.linear5 = nn.Linear(256, 256)
        torch.nn.init.xavier_uniform_(self.linear5.weight)
        self.linear6 = nn.Linear(256, 64)
        torch.nn.init.xavier_uniform_(self.linear6.weight)
        self.linear7 = nn.Linear(64, 16)
        torch.nn.init.xavier_uniform_(self.linear7.weight)
        self.linear8 = nn.Linear(16, 3)
        torch.nn.init.xavier_uniform_(self.linear8.weight)

    def forward(self, inp):
        x = inp
        x = relu(self.linear1(x))
        x = relu(self.linear2(x))
        x = relu(self.linear3(x))
        x = relu(self.linear4(x))
#        x = self.bn1(x)
        x = self.dropout(x)
        x = relu(self.linear5(x))
        x = relu(self.linear6(x))
        x = relu(self.linear7(x))
        outp = tanh(self.linear8(x))
        return (outp)
        
# data creation
vectors = None # put the vector variable name here - should be np.array of [n,6] shape
vectors2 = vectors[:,3:]/vectors[:,:3]

# loac model
model_path = 'd:/'
v2x_model = get_submodel()
v2x_model.load_state_dict(torch.load(model_path+'v2x_model.h5'))

# run model
xyz = v2x_model(vectors)

