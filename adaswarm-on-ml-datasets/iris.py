
# Common Imports

import time
import numpy as np
from torch.utils.data import Dataset, DataLoader
from keras.utils import to_categorical
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torchsummary import summary


from model import Model
from rempso import RotatedEMParicleSwarmOptimizer



class BCELossWithPSO(torch.autograd.Function):  
    @staticmethod
    def forward(ctx , y, y_pred, sum_cr, eta, gbest):
        ctx.save_for_backward(y, y_pred)
        ctx.sum_cr = sum_cr
        ctx.eta = eta
        ctx.gbest = gbest
        return F.binary_cross_entropy(y,y_pred)

    @staticmethod
    def backward(ctx, grad_output):
        yy, yy_pred= ctx.saved_tensors
        sum_cr = ctx.sum_cr
        eta = ctx.eta
        grad_input = torch.neg((sum_cr/eta) * (ctx.gbest - yy))
        return grad_input, grad_output, None, None, None


class BCELoss:
    def __init__(self, y):
        self.y = y
        self.fitness = torch.nn.BCELoss()
    def evaluate(self, x):
        # print(x, self.y)
        return self.fitness(x, self.y)
    
class PrepareData(Dataset):

    def __init__(self, X, y):
        if not torch.is_tensor(X):
            self.X = torch.from_numpy(X)
        if not torch.is_tensor(y):
            self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]



batch_size = 50
swarm_size = 10
num_epochs = 40



from sklearn.datasets import load_iris

#Load Iris
iris = load_iris()
X=iris.data
y=iris.target
y = to_categorical(y)

ds = PrepareData(X=X, y=y)
ds = DataLoader(ds, batch_size=batch_size, shuffle=True)



num_batches = int(X.shape[0]/ batch_size)
print(num_batches)

bce_loss = torch.nn.BCELoss()
proposed_loss = BCELossWithPSO.apply


optimizer_test = 'SGD'
model = Model(n_features=4, n_neurons=10, n_out=3)
optimizer = torch.optim.SGD(params=model.parameters(), lr=0.1)

sgd_losses = []
sgd_accuracy=[]
for e in range(num_epochs):
    batch_losses=[]
    accs = []
    for ix, (_x, _y) in enumerate(ds):
        
        #=========make inpur differentiable=======================
        tic = time.monotonic()
        _x = Variable(_x).float()
        _y = Variable(_y, ).float()
        #========forward pass=====================================
        yhat = model(_x).float()
        
        # print("==========================")
        loss = bce_loss(yhat, _y)
        acc = torch.eq(yhat.round(), _y).float().mean()# accuracy

        #=======backward pass=====================================
        optimizer.zero_grad() # zero the gradients on each pass before the update
        loss.backward() # backpropagate the loss through the model
        optimizer.step() # update the gradients w.r.t the loss

        accs.append(acc.item())
        toc = time.monotonic()
        batch_losses.append(loss.item())
        print("Batch : {}| Loss: {} | Time: {}".format(ix, loss.item(), toc-tic))
    sgd_losses.append(sum(batch_losses) / num_batches)
    sgd_accuracy.append(100 * sum(accs) / num_batches)
    if e % 1 == 0:
        print("[{}/{}], loss: {} acc: {}".format(e,
                                                 num_epochs, np.round(sum(batch_losses) / num_batches, 3),
                                                 100 * np.round(sum(accs) / num_batches, 3)))

print(sgd_losses)
print(sgd_accuracy)



optimizer_test = 'SGD-PSO'
model = Model(n_features=4, n_neurons=10, n_out=3)
optimizer = torch.optim.SGD(params=model.parameters(), lr=0.1)

pso_losses = []
pso_accuracy=[]
for e in range(num_epochs):
    batch_losses=[]
    accs = []
    for ix, (_x, _y) in enumerate(ds):
        
        #=========make inpur differentiable=======================
        tic = time.monotonic()
        _x = Variable(_x).float()
        _y = Variable(_y, ).float()
        _y.requires_grad = False
        p = RotatedEMParicleSwarmOptimizer(batch_size, swarm_size, 3, _y)
        p.optimize(BCELoss(_y))
        #========forward pass=====================================
        yhat = model(_x).float()
        for i in range(40):
            c1r1, c2r2, gbest = p.run_one_iter(verbosity=False)
        #print(gbest)
        # print("==========================")
        loss = proposed_loss(yhat, _y, c1r1+c2r2, 0.5, gbest)
        acc = torch.eq(yhat.round(), _y).float().mean()# accuracy

        #=======backward pass=====================================
        optimizer.zero_grad() # zero the gradients on each pass before the update
        loss.backward() # backpropagate the loss through the model
        optimizer.step() # update the gradients w.r.t the loss

        accs.append(acc.item())
        toc = time.monotonic()
        batch_losses.append(loss.item())
        print("Batch : {}| Loss: {} | Time: {}".format(ix, loss.item(), toc-tic))
    pso_losses.append(sum(batch_losses) / num_batches)
    pso_accuracy.append(100 * sum(accs) / num_batches)
    if e % 1 == 0:
        print("[{}/{}], loss: {} acc: {}".format(e,
                                                 num_epochs, np.round(sum(batch_losses) / num_batches, 3),
                                                 100 * np.round(sum(accs) / num_batches, 3)))

print(pso_losses)
print(pso_accuracy)


optimizer_test = 'RMSprop'
model = Model(n_features=4, n_neurons=10, n_out=3)
optimizer = torch.optim.RMSprop(params=model.parameters(), lr=0.1)

rms_losses = []
rms_accuracy=[]
for e in range(num_epochs):
    batch_losses=[]
    accs = []
    for ix, (_x, _y) in enumerate(ds):
        
        #=========make inpur differentiable=======================
        tic = time.monotonic()
        _x = Variable(_x).float()
        _y = Variable(_y, ).float()
        #========forward pass=====================================
        yhat = model(_x).float()
        
        # print("==========================")
        loss = bce_loss(yhat, _y)
        acc = torch.eq(yhat.round(), _y).float().mean()# accuracy

        #=======backward pass=====================================
        optimizer.zero_grad() # zero the gradients on each pass before the update
        loss.backward() # backpropagate the loss through the model
        optimizer.step() # update the gradients w.r.t the loss

        accs.append(acc.item())
        toc = time.monotonic()
        batch_losses.append(loss.item())
        print("Batch : {}| Loss: {} | Time: {}".format(ix, loss.item(), toc-tic))
    rms_losses.append(sum(batch_losses) / num_batches)
    rms_accuracy.append(100 * sum(accs) / num_batches)
    if e % 1 == 0:
        print("[{}/{}], loss: {} acc: {}".format(e,
                                                 num_epochs, np.round(sum(batch_losses) / num_batches, 3),
                                                 100 * np.round(sum(accs) / num_batches, 3)))

print(rms_losses)
print(rms_accuracy)




optimizer_test = 'Adam'
model = Model(n_features=4, n_neurons=10, n_out=3)
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.1)

adam_losses = []
adam_accuracy=[]
for e in range(num_epochs):
    batch_losses=[]
    accs = []
    for ix, (_x, _y) in enumerate(ds):
        
        #=========make inpur differentiable=======================
        tic = time.monotonic()
        _x = Variable(_x).float()
        _y = Variable(_y, ).float()
        #========forward pass=====================================
        yhat = model(_x).float()
        
        # print("==========================")
        loss = bce_loss(yhat, _y)
        acc = torch.eq(yhat.round(), _y).float().mean()# accuracy

        #=======backward pass=====================================
        optimizer.zero_grad() # zero the gradients on each pass before the update
        loss.backward() # backpropagate the loss through the model
        optimizer.step() # update the gradients w.r.t the loss

        accs.append(acc.item())
        toc = time.monotonic()
        batch_losses.append(loss.item())
        print("Batch : {}| Loss: {} | Time: {}".format(ix, loss.item(), toc-tic))
    adam_losses.append(sum(batch_losses) / num_batches)
    adam_accuracy.append(100 * sum(accs) / num_batches)
    if e % 1 == 0:
        print("[{}/{}], loss: {} acc: {}".format(e,
                                                 num_epochs, np.round(sum(batch_losses) / num_batches, 3),
                                                 100 * np.round(sum(accs) / num_batches, 3)))

print(adam_losses)
print(adam_accuracy)


optimizer_test = 'AMSGRad'
model = Model(n_features=4, n_neurons=10, n_out=3)
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.1, amsgrad=True)

amsgrad_losses = []
amsgrad_accuracy=[]
for e in range(num_epochs):
    batch_losses=[]
    accs = []
    for ix, (_x, _y) in enumerate(ds):
        
        #=========make inpur differentiable=======================
        tic = time.monotonic()
        _x = Variable(_x).float()
        _y = Variable(_y, ).float()
        #========forward pass=====================================
        yhat = model(_x).float()
        
        # print("==========================")
        loss = bce_loss(yhat, _y)
        acc = torch.eq(yhat.round(), _y).float().mean()# accuracy

        #=======backward pass=====================================
        optimizer.zero_grad() # zero the gradients on each pass before the update
        loss.backward() # backpropagate the loss through the model
        optimizer.step() # update the gradients w.r.t the loss

        accs.append(acc.item())
        toc = time.monotonic()
        batch_losses.append(loss.item())
        print("Batch : {}| Loss: {} | Time: {}".format(ix, loss.item(), toc-tic))
    amsgrad_losses.append(sum(batch_losses) / num_batches)
    amsgrad_accuracy.append(100 * sum(accs) / num_batches)
    if e % 1 == 0:
        print("[{}/{}], loss: {} acc: {}".format(e,
                                                 num_epochs, np.round(sum(batch_losses) / num_batches, 3),
                                                 100 * np.round(sum(accs) / num_batches, 3)))

print(amsgrad_losses)
print(amsgrad_accuracy)




optimizer_test = 'AdaGrad'
model = Model(n_features=4, n_neurons=10, n_out=3)
optimizer = torch.optim.Adagrad(params=model.parameters(), lr=0.1)

adagrad_losses = []
adagrad_accuracy=[]
for e in range(num_epochs):
    batch_losses=[]
    accs = []
    for ix, (_x, _y) in enumerate(ds):
        
        #=========make inpur differentiable=======================
        tic = time.monotonic()
        _x = Variable(_x).float()
        _y = Variable(_y, ).float()
        #========forward pass=====================================
        yhat = model(_x).float()
        
        # print("==========================")
        loss = bce_loss(yhat, _y)
        acc = torch.eq(yhat.round(), _y).float().mean()# accuracy

        #=======backward pass=====================================
        optimizer.zero_grad() # zero the gradients on each pass before the update
        loss.backward() # backpropagate the loss through the model
        optimizer.step() # update the gradients w.r.t the loss

        accs.append(acc.item())
        toc = time.monotonic()
        batch_losses.append(loss.item())
        print("Batch : {}| Loss: {} | Time: {}".format(ix, loss.item(), toc-tic))
    adagrad_losses.append(sum(batch_losses) / num_batches)
    adagrad_accuracy.append(100 * sum(accs) / num_batches)
    if e % 1 == 0:
        print("[{}/{}], loss: {} acc: {}".format(e,
                                                 num_epochs, np.round(sum(batch_losses) / num_batches, 3),
                                                 100 * np.round(sum(accs) / num_batches, 3)))

print(adagrad_losses)
print(adagrad_accuracy)



optimizer_test = 'Adadelta'
model = Model(n_features=4, n_neurons=10, n_out=3)
optimizer = torch.optim.Adadelta(params=model.parameters(), lr=0.1)

adadelta_losses = []
adadelta_accuracy=[]
for e in range(num_epochs):
    batch_losses=[]
    accs = []
    for ix, (_x, _y) in enumerate(ds):
        
        #=========make inpur differentiable=======================
        tic = time.monotonic()
        _x = Variable(_x).float()
        _y = Variable(_y, ).float()
        #========forward pass=====================================
        yhat = model(_x).float()
        
        # print("==========================")
        loss = bce_loss(yhat, _y)
        acc = torch.eq(yhat.round(), _y).float().mean()# accuracy

        #=======backward pass=====================================
        optimizer.zero_grad() # zero the gradients on each pass before the update
        loss.backward() # backpropagate the loss through the model
        optimizer.step() # update the gradients w.r.t the loss

        accs.append(acc.item())
        toc = time.monotonic()
        batch_losses.append(loss.item())
        print("Batch : {}| Loss: {} | Time: {}".format(ix, loss.item(), toc-tic))
    adadelta_losses.append(sum(batch_losses) / num_batches)
    adadelta_accuracy.append(100 * sum(accs) / num_batches)
    if e % 1 == 0:
        print("[{}/{}], loss: {} acc: {}".format(e,
                                                 num_epochs, np.round(sum(batch_losses) / num_batches, 3),
                                                 100 * np.round(sum(accs) / num_batches, 3)))

print(adadelta_losses)
print(adadelta_accuracy)



optimizer_test = 'AdaSwarm'
model = Model(n_features=4, n_neurons=10, n_out=3)
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.1)

adaswarm_losses = []
adaswarm_accuracy=[]
for e in range(num_epochs):
    batch_losses=[]
    accs = []
    for ix, (_x, _y) in enumerate(ds):
        
        #=========make inpur differentiable=======================
        tic = time.monotonic()
        _x = Variable(_x).float()
        _y = Variable(_y, ).float()
        _y.requires_grad = False
        p = RotatedEMParicleSwarmOptimizer(batch_size, swarm_size, 3, _y)
        p.optimize(BCELoss(_y))
        #========forward pass=====================================
        yhat = model(_x).float()
        for i in range(40):
            c1r1, c2r2, gbest = p.run_one_iter(verbosity=False)
        #print(gbest)
        # print("==========================")
        loss = proposed_loss(yhat, _y, c1r1+c2r2, 0.1, gbest)
        acc = torch.eq(yhat.round(), _y).float().mean()# accuracy

        #=======backward pass=====================================
        optimizer.zero_grad() # zero the gradients on each pass before the update
        loss.backward() # backpropagate the loss through the model
        optimizer.step() # update the gradients w.r.t the loss

        accs.append(acc.item())
        toc = time.monotonic()
        batch_losses.append(loss.item())
        print("Batch : {}| Loss: {} | Time: {}".format(ix, loss.item(), toc-tic))
    adaswarm_losses.append(sum(batch_losses) / num_batches)
    adaswarm_accuracy.append(100 * sum(accs) / num_batches)
    if e % 1 == 0:
        print("[{}/{}], loss: {} acc: {}".format(e,
                                                 num_epochs, np.round(sum(batch_losses) / num_batches, 3),
                                                 100 * np.round(sum(accs) / num_batches, 3)))

print(adaswarm_losses)
print(adaswarm_accuracy)




import matplotlib.pyplot as plt
dataset_name = "Iris Dataset"

plt.figure(figsize=(20,10))
plt.title(dataset_name + " Loss")
plt.plot(adam_losses, label="Adam")
plt.plot(adaswarm_losses, label="AdaSwarm")
plt.plot(rms_losses, label="RMSProp")
plt.plot(sgd_losses, label="SGD")
plt.plot(pso_losses, label="Emulate SGD-PSO Params")
plt.legend()
plt.show()


# In[41]:


plt.figure(figsize=(20,10))
plt.title(dataset_name + " Accuracy")
plt.plot(adam_accuracy, label="Adam")
plt.plot(adaswarm_accuracy, label="AdaSwarm")
plt.plot(rms_accuracy, label="RMSProp")
plt.plot(sgd_accuracy, label="SGD")
plt.plot(pso_accuracy, label="Emulate SGD-PSO Params")
plt.legend()
plt.show()


# In[ ]:




