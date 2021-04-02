# DogsVsCats
Algorithm competition organized by Kaggle, used for PyTorch initial study

## Dataset
Provided from Microsoft Research: https://www.kaggle.com/c/dogs-vs-cats/data

## Convolutional network design
![CNN](https://github.com/Ericdiii/DogsVsCats-PyTorch-CNN/blob/main/CNN.png)
1. **Input**: Adjust image to 200×200 pixel
2. **ConV1**: The scale of convolutional core is (3×3×3×16), hight=3, width=3, #layer=3, #filters=16
3. **Result of first convolution**: (200×200×16) feature map
4. **Pooling**: 2×2 Max pooling
5. **Result of first pooling**: 
6. **ConV2**: The scale of convolutional core is (3×3×16×16), hight=3, width=3, #layer=16, #filters=16
7. **Result of second convolution**: (100×100×16) feature map
8. **Pooling**: 2×2 Max pooling
9. 
