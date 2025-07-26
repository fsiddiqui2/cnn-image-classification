from tqdm import tqdm
import numpy as np
import pandas as pd
from torchvision import transforms
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torchvision import transforms
import sklearn.metrics as metrics

import json
import os
import sys
import argparse

from model import Model
from dataset import CatDogDataset

from torchmetrics import Accuracy, F1Score, AUROC


def main(args):
    def test(model, dataloader, criterion):
        model.eval()
        running_loss = 0
        accuracy = Accuracy(task="binary").to(device)
        f1 = F1Score(task="binary").to(device)
        auroc = AUROC(task="binary").to(device)
        
        with torch.no_grad():
            for features, target in tqdm(dataloader):
                features, target = features.to(device), target.to(device)
                output = model(features).squeeze() #[32, 1] -> [32] to match with labels
                
                target = target.float()
                loss = criterion(output, target)
                running_loss += loss.item() * features.size(0)

                accuracy(output, target)
                f1(output, target)
                auroc(output, target)

        epoch_loss = running_loss/len(dataloader.dataset)
        epoch_acc = accuracy.compute().item()
        epoch_f1 = f1.compute().item()
        epoch_auroc = auroc.compute().item()

        return epoch_loss, epoch_acc, epoch_f1, epoch_auroc
        
    image_dim = 224
    transform = transforms.Compose([
        transforms.Resize([image_dim, image_dim]),
        transforms.ToTensor(),
    ])

    # save_dir = "CNN2_lr1e-4_bs128_dropout"
    save_dir = args.save_dir
    data_dir = "PetImages"

    testset = CatDogDataset(split="test", root_dir=data_dir, transform=transform)

    batch_size = args.batch_size
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)

    # model = Model(image_dim=224)
    # model = model.to(device)
    # model.load_state_dict(torch.load(os.path.join(save_dir, "best_model_weights.pth")))
    model = torch.load(os.path.join(save_dir, "best_model.pth"), weights_only = False)

    criterion = nn.BCEWithLogitsLoss()
    criterion = criterion.to(device)



    print("-----Testing Model-----")
    test_loss, test_acc, test_f1, test_roc = test(model, testloader, criterion)
    print(f"Test Loss: {test_loss}")
    print(f"Test Acc: {test_acc}")
    
    with open(os.path.join(save_dir, "results.json"), "w") as f:
        json.dump(
            {
                "test_loss" : test_loss,
                "test_acc" : test_acc,
                "test_f1" : test_f1, 
                "test_roc" : test_roc,
            }, f, indent=4)
    

if __name__ == "__main__":
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("MPS device found.")
    elif torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("CUDA device found.")
    else:
        device = torch.device("cpu")
        print("MPS or CUDA device not found. Using CPU.")

    parser = argparse.ArgumentParser(description="Dog Cat CNN Classifier Testing Script")
    parser.add_argument("--save-dir", type=str, help="Directory to fetch model and save results")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for testing")
    parser.add_argument("--data-seed", type=int, default=42, help="Random seed for train-val-test split")


    args = parser.parse_args()

    main(args)
    print("Finished")

