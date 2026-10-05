from torch.utils.data import DataLoader

from hiperparametro import *
from rede_dataset import AlgebraicDataSet

line = lambda x: 2*x + 3
interval = (-10, 10)
train_nsamples = 1000
test_nsamples = 100


train_dataset = AlgebraicDataSet(line, interval, train_nsamples)
test_dataset = AlgebraicDataSet(line, interval, test_nsamples)

train_dataloader = DataLoader(train_dataset, batch_size=train_nsamples, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=test_nsamples, shuffle=True)


def train(model, dataLoader, lossfunc, optimizer):

    cumloss = 0.0
    model.train()

    for X, y in dataLoader:
        X = X.unsqueeze(1).float().to(device)
        y = y.unsqueeze(1).float().to(device)
        #[ [1], [2], [3], [4] ]

        pred = model(X)
        loss = lossfunc(pred, y)

        #zera os gradientes acumulados
        optimizer.zero_grad()
        #computa os gradientes
        loss.backward()
        #anda na direção que reduz o erro local
        optimizer.step()

        #loss é um tensor; item para obter o float
        cumloss += loss.item()

    return cumloss / len(dataLoader)


def test(model, dataLoader, lossfunc):
    cumloss = 0.0
    model.eval()
    with torch.no_grad():
        for X, y in dataLoader:
            X = X.unsqueeze(1).float().to(device)
            y = y.unsqueeze(1).float().to(device)
            # [ [1], [2], [3], [4] ]

            pred = model(X)
            loss = lossfunc(pred, y)

            cumloss += loss.item()

    return cumloss / len(dataLoader)

def plot_comparinson(f, model, interval=(-10, 10), nsamples=10):
  fig, ax = plt.subplots(figsize=(10, 10))

  ax.grid(True, which='both')
  ax.spines['left'].set_position('zero')
  ax.spines['right'].set_color('none')
  ax.spines['bottom'].set_position('zero')
  ax.spines['top'].set_color('none')

  samples = np.linspace(interval[0], interval[1], nsamples)
  model.eval()
  with torch.no_grad():
    pred = model(torch.tensor(samples).unsqueeze(1).float().to(device))

  ax.plot(samples, list(map(f, samples)), "o", label="ground truth")
  ax.plot(samples, pred.cpu(), label="model")
  plt.legend()
  plt.show()


epochs = 201
for t in range(epochs):
  print("vai!")
  train_loss = train(model, train_dataloader, lossfunc, optimizer)
  if t % 10 == 0:
    print(f"Epoch: {t}; Train Loss: {train_loss}")
    plot_comparinson(line, model)

test_loss = test(model, test_dataloader, lossfunc)
print(f"Test Loss: {test_loss}")