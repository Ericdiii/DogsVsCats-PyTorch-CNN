# Dogs Vs Cats
Kaggle algorithm competition topic </br>
It's a good practice for studying CNN and PyTorch :)

## Dataset
Provided from Microsoft Research: https://www.kaggle.com/c/dogs-vs-cats/data

## Convolutional network design
<img src="https://github.com/Ericdiii/DogsVsCats-PyTorch-CNN/blob/main/CNN.png" height="260"/>

1. **Input**: Adjust image to `200×200` pixels
2. **ConV1**: The scale of convolutional core is `(3×3×3×16)`, hight=`3`, width=`3`, #layer=`3`, #filters=`16`
3. **Result of first convolution**: `(200×200×16)` feature map
4. **Pooling**: `2×2` Max pooling
5. **Result of first pooling**: The image is reduced to `100×100` pixels
6. **ConV2**: The convolutional kernel is `(3×3×16×16)`, hight=`3`, width=`3`, #layer=`16`, #filters=`16`
7. **Result of second convolution**: `(100×100×16)` feature map
8. **Pooling**: `2×2` Max pooling
9. **Result of second pooling**: The image is reduced to `50×50` pixels
10. **FC1**: 50×50×16=`40000` input nodes, `128` output nodes, output data is `(128×1)`
11. **FC2**: `128` input nodes, `64` output nodes, output data is `(64×1)`
12. **FC3**: `64` input nodes, `2` output nodes to represent the percentage of cat or dog (Softmax)

## Usage

1. Download the dataset
2. Train the image classification model
```sh
train_CNN.py
```
3. Test the trained model
```sh
test_CNN.py
```

<img src="https://github.com/Ericdiii/DogsVsCats-PyTorch-CNN/blob/main/TestOutput1.png" height="260"/> <img src="https://github.com/Ericdiii/DogsVsCats-PyTorch-CNN/blob/main/TestOutput2.png" height="280"/> 


## References
- **DogsVsCats**  https://github.com/xbliuHNU/DogsVsCats</br>
- **DogsVsCats-ResNet18**  https://github.com/xbliuHNU/DogsVsCats-ResNet18
