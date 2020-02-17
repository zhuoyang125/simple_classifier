# Simple Classifier
This repo would allow you to train a image classifier using Tensorflow Keras. The model architecture is based of SimpleNet. https://arxiv.org/abs/1608.06037

## Setup

## How to train
Go to `config` to adjust your training parameters. Choice of MNIST or CIFAR10 dataset for training, and choice of data augmentation. Use help function to view arg options. To train custom dataset, include 'data' folder with directory structure:
```
data/
	train/
		class_a/
			class_a01.jpg
			class_a02.jpg
			...
		class_b/
			class_b01.jpg
			class_b02.jpg
			...
	test/
		class_a/
			class_a01.jpg
			class_a02.jpg
			...
		class_b/
			class_b01.jpg
			class_b02.jpg
```
Command
```bash
python training.py
```

## Evaluation
Load saved model by specifying model path in argument. Also allows loading of custom dataset and data augmentation.
```bash
python evaluation.py
```

