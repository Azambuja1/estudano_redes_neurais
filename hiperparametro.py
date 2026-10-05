import numpy as np

from rede_dataset import AlgebraicDataSet
from rede_simples import LineNetwork
import torch
from torch import nn
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"rodando na: {device}")

model = LineNetwork()

#Função de perda (loss function)
#Erro quadrático médio (Mean Squared Error)
lossfunc = nn.MSELoss()

#Grandiente Descendente Estocásticom
#SGB = Stochastic Gradient Descent
optimizer = torch.optim.SGD(model.parameters(), lr = 1e-3)

#taxa de aprendizado lr = learning rate
