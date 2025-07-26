from tqdm import tqdm
import numpy as np
import pandas as pd
from torchvision import transforms
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
# from torchvision import transforms
import torchvision.transforms.v2 as transforms
import sklearn.metrics as metrics

import json
import os
import sys
import argparse

from model import Model
from dataset import CatDogDataset

from torchmetrics import Accuracy, F1Score, AUROC


def main(args):
    def train(model, dataloader, criterion, optimizer):
        model.train()
        running_loss = 0
        accuracy = Accuracy(task="binary").to(device)
        f1 = F1Score(task="binary").to(device)
        auroc = AUROC(task="binary").to(device)

        for features, target in tqdm(dataloader):
            features, target = features.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(features).squeeze() #[32, 1] -> [32] to match with labels

            target = target.float()
            loss = criterion(output, target)
            running_loss += loss.item() * features.size(0)
            loss.backward()
            optimizer.step()

            accuracy(output, target)
            f1(output, target)
            auroc(output, target)
       
        epoch_loss = running_loss/len(dataloader.dataset)
        epoch_acc = accuracy.compute().item()
        epoch_f1 = f1.compute().item()
        epoch_auroc = auroc.compute().item()

        return epoch_loss, epoch_acc, epoch_f1, epoch_auroc

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
    train_transform = transforms.Compose([
        transforms.Resize([image_dim, image_dim]),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
    ])
    transform = transforms.Compose([
        transforms.Resize([image_dim, image_dim]),
        transforms.ToTensor(),
    ])

    data_dir = "PetImages"
    data_seed = args.data_seed
    trainset = CatDogDataset(split="train", root_dir=data_dir, transform=train_transform, random_seed=data_seed)
    valset = CatDogDataset(split="val", root_dir=data_dir, transform=transform, random_seed=data_seed)
    testset = CatDogDataset(split="test", root_dir=data_dir, transform=transform, random_seed=data_seed)

    print(f"Percent Positive Class - Train: {trainset.class_distribution():.4f}, Val: {valset.class_distribution():.4f}, Test: {testset.class_distribution():.4f}")

    batch_size = args.batch_size
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
    valloader = DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=0)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)

    lr = args.lr
    model = Model(image_dim=224)
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = criterion.to(device)

    train_history = {
        "train_loss": [],
        "train_acc" : [],
        "train_f1" : [],
        "train_roc" : [],
        "val_loss" : [],
        "val_acc" : [],
        "val_f1" : [],
        "val_roc" : [],
    }

    # save_dir = "CNN2_lr1e-4_bs128_dropout"
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok = True)

    best_val_loss = np.inf
    epochs = args.epochs
    for epoch in range(1, epochs+1):
        train_loss, train_acc, train_f1, train_roc = train(model, trainloader, criterion, optimizer)
        val_loss, val_acc, val_f1, val_roc = test(model, valloader, criterion)

        print(f"-----Epoch {epoch}-----")
        print(f"Train Loss: {train_loss}, Train Acc: {train_acc}")
        print(f"Val Loss: {val_loss}, Val Acc: {val_acc}")
        
        train_history["train_loss"].append(train_loss)
        train_history["train_acc"].append(train_acc)
        train_history["train_f1"].append(train_f1)
        train_history["train_roc"].append(train_roc)

        train_history["val_loss"].append(val_loss)
        train_history["val_acc"].append(val_acc)
        train_history["val_f1"].append(val_f1)
        train_history["val_roc"].append(val_roc)

        with open(os.path.join(save_dir, "train_history.json"), "w") as f:
            json.dump(train_history, f, indent=4)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print("Saving best model...")
            # torch.save(model.state_dict(), os.path.join(save_dir,"best_model_weights.pth"))
            torch.save(model, os.path.join(save_dir,"best_model.pth"))

    print("Saving final model...")     
    torch.save(model, os.path.join(save_dir,"final_model.pth"))

    print("-----Testing Model-----")
    # model = torch.load(os.path.join(save_dir, "best_model_weights.pth"), weights_only = False)
    test_loss, test_acc, test_f1, test_roc = test(model, testloader, criterion)
    print(f"Test Loss: {test_loss}")
    print(f"Test Acc: {test_acc}")
    
    with open(os.path.join(save_dir, "end_results.json"), "w") as f:
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


    parser = argparse.ArgumentParser(description="Dog Cat CNN Classifier")
    parser.add_argument("--save-dir", type=str, help="Directory to save model, history, and results")
    parser.add_argument("--lr", type=float, default="1e-4", help="Model learning rate")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for training")
    parser.add_argument("--data-seed", type=int, default=42, help="Random seed for train-val-test split")
    parser.add_argument("--name", type=str, help="If save-dir not specified, will be set to [name]_lr[lr]_bs[batch-size]")

    args = parser.parse_args()

    if args.save_dir == None:
        args.save_dir = f"{args.name}_lr{args.lr}_bs{args.batch_size}_ep{args.epochs}_seed{args.data_seed}"


    main(args)
    print("Finished")

