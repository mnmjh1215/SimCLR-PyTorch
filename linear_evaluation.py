
from models.resnet import ResNet18, ResNet50
from models.projection import ProjectionHead
from models.simclr_model import SimCLRModel

from dataloader import load_cifar10_for_linear_evaluation

from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm


def train_on_PCA(pca_dim=512):
    cifar10_train_dataloader, cifar10_test_dataloader = load_cifar10_for_linear_evaluation()
    
    # prepare train and test data
    print("Preparing CIFAR10 Data")
    train_X = []
    train_Y = []
    for x, y in tqdm(cifar10_train_dataloader):
        train_X.append(x.view(x.size(0), -1).numpy())
        train_Y.append(y.numpy())
        
    train_X = np.concatenate(train_X, axis=0)  # (N, 32*32*3)
    train_Y = np.concatenate(train_Y, axis=0)  # (N, )

    test_X = []
    test_Y = []
    for x, y in tqdm(cifar10_test_dataloader):
        test_X.append(x.view(x.size(0), -1).numpy())
        test_Y.append(y.numpy())
        
    test_X = np.concatenate(test_X, axis=0)  # (N, 32*32*3)
    test_Y = np.concatenate(test_Y, axis=0)  # (N, )
    
    print("train data shape:", train_X.shape)
    print("test data shape:", test_X.shape)
    
    # use PCA to extract features
    print("Extracting Features using PCA")
    pca = PCA(n_components=pca_dim)
    
    train_X_pca = pca.fit_transform(train_X)
    test_X_pca = pca.transform(test_X)
    
    print("train data shape after PCA:", train_X_pca.shape)
    print("test data shape after PCA:", test_X_pca.shape)
    
    # train linear model
    print("Training Linear Model")
    lr_clf = LogisticRegression(solver='lbfgs', multi_class='multinomial', max_iter=1000, verbose=1)
    lr_clf.fit(train_X_pca, train_Y)
    
    print("Train Accuracy", lr_clf.score(train_X_pca, train_Y))
    print("Test Accuracy", lr_clf.score(test_X_pca, test_Y))
    
    
def train_on_resnet_features(args):
    
    print("Creating Model")
    if args.model_type == 'ResNet18':
        resnet = ResNet18()
        projection = ProjectionHead(512, 512, 512)
    elif args.model_type == 'ResNet50':
        resnet = ResNet50()
        projection = ProjectionHead(2048, 2048, 2048)
    else:
        raise NotImplementedError
    model = SimCLRModel(resnet, projection)
    
    print("Loading Checkpoint")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    model.to(device)
    model.eval()
    
    print("Loading CIFAR10 data")
    cifar10_train_dataloader, cifar10_test_dataloader = load_cifar10_for_linear_evaluation(batch_size=16)

    print("Encoding CIFAR10 data")
    train_X = []
    train_Y = []
    for x, y in tqdm(cifar10_train_dataloader):
        x = x.to(device)
        train_X.append(model.encode(x).detach().cpu().numpy())
        train_Y.append(y.numpy())
        
    train_X = np.concatenate(train_X, axis=0)  # (N, D)
    train_Y = np.concatenate(train_Y, axis=0)  # (N, )

    test_X = []
    test_Y = []
    for x, y in tqdm(cifar10_test_dataloader):
        x = x.to(device)
        test_X.append(model.encode(x).detach().cpu().numpy())
        test_Y.append(y.numpy())
        
    test_X = np.concatenate(test_X, axis=0)  # (N, D)
    test_Y = np.concatenate(test_Y, axis=0)  # (N, )
    
    print("train data shape:", train_X.shape)
    print("test data shape:", test_X.shape)
    
    # train linear model
    print("Training Linear Model")
    lr_clf = LogisticRegression(solver='lbfgs', multi_class='multinomial', max_iter=args.max_iter, verbose=1)
    lr_clf.fit(train_X, train_Y)
    
    print("Train Accuracy", lr_clf.score(train_X, train_Y))
    print("Test Accuracy", lr_clf.score(test_X, test_Y))


def get_args():
    import argparse
    
    argument_parser = argparse.ArgumentParser()
    
    argument_parser.add_argument("run_type",
                                 choices=["pca", "resnet"])
    
    argument_parser.add_argument("--checkpoint_path",
                                 default='checkpoints/resnet18-lr0.003-epochs250.pth')
    
    argument_parser.add_argument("--model_type",
                                 choices=['ResNet18', 'ResNet50'],
                                 default='ResNet18')
    
    argument_parser.add_argument("--max_iter",
                                 type=int,
                                 default=1500)
        
    args = argument_parser.parse_args()
    
    return args


if __name__ == '__main__':
    
    args = get_args()
    if args.run_type == 'pca':
        train_on_PCA()
    else:
        train_on_resnet_features(args)