from network import Net
import torch
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image
import random
import os
import data

dataset_dir = './data/test/'                    # Dataset path
model_file = './model/model.pth'                # Model path
N = 10

def test():

    # Setting model
    model = Net()                                       # Network instantiation
    model.cuda()                                        # Send network to the GPU for calculation
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(model_file))       # Load the trained model parameters
    model.eval()                                        # Set to evaluation mode: no dropout

    # Get data
    files = random.sample(os.listdir(dataset_dir), N)   # Randomly obtain N test images
    imgs = []           # img
    imgs_data = []      # img data
    
    for file in files:
        img = Image.open(dataset_dir + file)            # Open image
        img_data = data.dataTransform(img)              # Convert to Tensor data

        imgs.append(img)                                # Image list
        imgs_data.append(img_data)                      # Tensor list
    imgs_data = torch.stack(imgs_data)                  # 4D Tensor

    # Calculation
    out = model(imgs_data)                              # Calculate each image
    out = F.softmax(out, dim=1)                         # Output probability
    out = out.data.cpu().numpy()                        # Convert to numpy data

    # Print results 
    for idx in range(N):
        plt.figure()
        if out[idx, 0] > out[idx, 1]:
            plt.suptitle('cat:{:.1%},dog:{:.1%}'.format(out[idx, 0], out[idx, 1]))
        else:
            plt.suptitle('dog:{:.1%},cat:{:.1%}'.format(out[idx, 1], out[idx, 0]))
        plt.imshow(imgs[idx])
    plt.show()


if __name__ == '__main__':
    test()
