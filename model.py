import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

class TopologyNet(nn.Module):
    def __init__(self):
        super(TopologyNet,self).__init__()

        #Defining a single layer deep neural network

        #Input Layer
        self.fc1 = nn.Linear(784,100,bias=False)

        #Hidden Layer
        self.fc2 = nn.Linear(100,10,bias=False)

    def forward(self,x):
        out = F.sigmoid(self.fc1(x))
        out = F.sigmoid(self.fc2(out))
        out = F.log_softmax(out, dim=1)
        return out 


def train(model, device, train_loader, optimizer, epoch):
    f = open("demofile.txt", "w")
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        data = data.view(-1, 28*28).requires_grad_()
        optimizer.zero_grad()
        output = model(data)
        trained_weights = model.fc2.weight
        print(trained_weights.size())
        f.write(str(trained_weights.detach().numpy()))
        f.write(",")
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    f.close()

def main():
    device = torch.device("cpu")
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=64, shuffle=True)

    model = TopologyNet().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

    for epoch in range(1, 2):
        train(model, device, train_loader,optimizer,epoch)

if __name__ == "__main__":
    main()

