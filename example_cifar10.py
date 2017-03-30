import tensorflow as tf
import random
import os
import numpy as np
from convnet import ConvNet, load_yaml_config

config_path = 'config/cifar10.yaml'

config = load_yaml_config(config_path)
nets = ConvNet(config)

nets.train(max_steps=200000)
