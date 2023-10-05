# arg parse
import argparse
import torch
import os
# Example Input: python train.py flowers --save_dir model_saved
parser = argparse.ArgumentParser(description="Train Models")
parser.add_argument("dir_data", action="store", type=str)
# only for directory not files name
parser.add_argument("--save_dir", dest="directory_name", action="store", type=str, default="saved_models")
# for architecture
parser.add_argument("--arch", dest="arch", action="store", type=str,  default="vgg16")
# set hyperparameters 
parser.add_argument("--learning_rate", dest="learn_rate", type=float, default="0.001")
parser.add_argument("--hidden_units", dest="hidden_units", type=int, default="512")
parser.add_argument("--epochs", dest="epochs", type=int, default=10)
# train with gpu
parser.add_argument("--gpu", action="store_true")

result_parser = parser.parse_args()
# os.mkdir( result_parser.directory_name)
# torch.save("Mumumumu", result_parser.directory_name + "/helloo.txt")
print(result_parser.gpu)
print(result_parser.dir_data)
print(result_parser.epochs)
print(result_parser.directory_name)