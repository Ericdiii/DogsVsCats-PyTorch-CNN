from getdata import DogsVSCatsDataset as DVCD
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
        # 读取数据集中数据进行训练，因为dataloader的batch_size设置为16，所以每次读取的数据量为16，即img包含了16个图像，label有16个
        # 循环读取封装后的数据集，其实就是调用了数据集中的__getitem__()方法，只是返回数据格式进行了一次封装
        for img, label in dataloader:
            # 将数据放置在PyTorch的Variable节点中，并送入GPU中作为网络计算起点
            img, label = Variable(img).cuda(), Variable(label).cuda()
            # 计算网络输出值，就是输入网络一个图像数据，输出猫和狗的概率，调用了网络中的forward()方法
            out = model(img)
            # 计算损失，也就是网络输出值和实际label的差异，显然差异越小说明网络拟合效果越好，此处需要注意的是第二个参数，必须是一个1维Tensor
            loss = criterion(out, label.squeeze())
            # 误差反向传播，采用求导的方式，计算网络中每个节点参数的梯度，显然梯度越大说明参数设置不合理，需要调整
            loss.backward()
            # 优化采用设定的优化方法对网络中的各个参数进行调整
            optimizer.step()                            
            optimizer.zero_grad()
            # 清除优化器中的梯度以便下一次计算，因为优化器默认会保留，不清除的话，每次计算梯度都回累加
            cnt += 1

            # 打印一个batch size的训练结果
            print('Epoch:{0},Frame:{1}, train_loss {2}'.format(epoch, cnt*batch_size, loss/batch_size))          

    # 训练所有数据后，保存网络的参数
    torch.save(model.state_dict(), '{0}/model.pth'.format(model_cp))            


if __name__ == '__main__':
    train()
