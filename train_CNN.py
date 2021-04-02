from data import DogsVSCatsDataset as DVCD
from torch.utils.data import DataLoader as DataLoader
from network import Net
import torch
from torch.autograd import Variable
import torch.nn as nn

dataset_dir = './data/'             # Dataset path
model_cp = './model/'               # Save location of network parameters

workers = 10                        # Number of data thread read by PyTorch 
batch_size = 16                     # Batch size
lr = 0.0001                         # Learning rate
nepoch = 10                         # Number of epoch

def train():
    # Dataset instantiation
    datafile = DVCD('train', dataset_dir) 
    # Encapsulate using PyTorch's DataLoader class to achieve the disturbing the order of data, 
    # multi-threaded reading, and fetching multiple data at a time
    dataloader = DataLoader(datafile, batch_size=batch_size, shuffle=True, num_workers=workers, drop_last=True)     

    print('Dataset loaded! length of train set is {0}'.format(len(datafile)))

    # Network instantiation
    model = Net()
    # Send the network to the GPU to speed up calculations
    model = model.cuda()
    model = nn.DataParallel(model)
    
    # The network is set to training mode
    # There are two modes to choose from: training mode .train() and evaluation mode .eval()
    # The difference is training mode uses dropout to prevent network overfitting
    model.train()

    # Optimizer instantiation
    # Adam method to adjust network hyperparameters for optimization
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Use cross entropy to calculate loss: the result will smaller when the two values closer
    criterion = torch.nn.CrossEntropyLoss() 

    # Number of training images
    cnt = 0             
    for epoch in range(nepoch):
        # Read the data from dataset. Each time can read 16 images due to the batch_size is set to 16
        # Loop reading the encapsulated dataset is calling the __getitem__() in the data.py, returning the data format and conduct an encapsulating
        for img, label in dataloader:
            # Place the data in the Variable node of PyTorch and send it to the GPU as the starting point for network calculations
            img, label = Variable(img).cuda(), Variable(label).cuda()
            # Calculate the network output value: Input image data, call forward() to output the probability of cat or dog
            out = model(img)
            # Calculate the loss. The second parameter must be a 1-dimensional Tensor
            loss = criterion(out, label.squeeze())
            # Back propagation: Calculate the gradient of each node by derivation
            # The larger the gradient, the unreasonable parameter setting and need to be adjusted
            loss.backward()
            # Optimize by adjusting the parameters
            optimizer.step()
            # Clear the gradient in the optimizer for the next calculation
            optimizer.zero_grad()
            
            cnt += 1

            # Print the training results of each batch size
            print('Epoch:{0},Frame:{1}, train_loss {2}'.format(epoch, cnt*batch_size, loss/batch_size))          

    # Save the network parameters after training all data
    torch.save(model.state_dict(), '{0}/model_CNN.pth'.format(model_cp))            

if __name__ == '__main__':
    train()
