from tqdm import tqdm
from torch.autograd import Variable
import torch
import torch.nn as nn
import os
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from transformers import AutoImageProcessor, AutoModelForImageClassification, Trainer, TrainingArguments, AutoConfig
from data_utils import split_dataset, MusicDataset, get_dataloaders
from model_utils import train_from_hf, plot_confusion_matrix
from torchvision import transforms
import argparse
import json
from sklearn.metrics import classification_report

class NoneTransform(object):
    ''' Does nothing to the image. To be used instead of None '''

    def __call__(self, image):
        return image

if __name__ == "__main__":

    args = argparse.ArgumentParser()
    args.add_argument("--data_path", type=str, default="genres")
    args.add_argument("--model_name", type=str, default="google/efficientnet-b0")
    args.add_argument("--cfg", type=str, default="./cfg/training_cfg.json")
    args.add_argument('--fake_rgb', action='store_true')
    args = args.parse_args()

    music_dataset = MusicDataset(os.path.join('data', args.data_path))

    config = AutoConfig.from_pretrained(args.model_name)
    config.num_labels = music_dataset.genres.shape[0]

    feature_extractor = AutoImageProcessor.from_pretrained(args.model_name)
    model = AutoModelForImageClassification.from_pretrained(args.model_name)

    print(model)
    if len(feature_extractor.size.values()) == 1:
        feature_extractor.size = {k: (int(v), int(v)) for k, v in feature_extractor.size.items()}

    model_transforms = transforms.Compose([
        # transforms.Resize(tuple(next(iter(feature_extractor.size.values())))),
        transforms.RandomHorizontalFlip(),  # Data augmentation
        # transforms.RandomCrop(size=tuple(next(iter(feature_extractor.size.values()))), padding=4),  # Data augmentation
        transforms.RandomRotation(degrees=15),  # Data augmentation
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)) if args.fake_rgb else NoneTransform(),
        transforms.Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)
    ])
    print(model_transforms)
    music_dataset.transform = model_transforms

    train_dataset, val_dataset, test_dataset = split_dataset(music_dataset)
    train_loader, val_loader, test_loader = get_dataloaders(music_dataset)

    with open(args.cfg, "r") as f:
        cfg = json.load(f)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    name = cfg['name']
    lr = cfg['lr']
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=cfg['weight_decay'])  # Added weight decay
    criterion = nn.CrossEntropyLoss().to(device)

    train_from_hf(model, image_processor, train_loader, val_loader, optimizer, criterion, device=device, **cfg)

    # Early stopping and best model saving logic
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0

    for epoch in range(cfg['num_epochs']):
        model.train()
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.logits, labels)
            loss.backward()
            optimizer.step()

        val_loss = 0.0
        val_acc = 0.0
        num_val_batches = len(val_loader)

        model.eval()
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs.logits, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.logits, 1)
                val_acc += (predicted == labels).sum().item()

        val_loss /= num_val_batches
        val_acc /= len(val_loader.dataset)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            #torch.save(model.state_dict(), os.path.join('ckpt', f'{name}_best_model.pth'))
            model.save_pretrained(os.path.join('ckpt',f'{name}_best_model_true'))
            image_processor.save_pretrained(os.path.join('ckpt',f'{name}_best_model_true'))
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping")
            break

        print(f'Epoch {epoch + 1}/{cfg["num_epochs"]}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

    model.load_state_dict(torch.load(os.path.join('ckpt', f'{name}_best_model.pth')))
    model.to(device)

    model.eval()
    all_preds = []
    all_labels = []
    test_acc = 0.0
    num_test_batches = len(test_loader)

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc='Testing'):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.logits, 1)
            all_preds.extend(predicted.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
            test_acc += (predicted == labels).sum().item()

    test_acc /= len(test_loader.dataset)

    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='micro')
    report = classification_report(all_labels, all_preds)
    print(report)
    print(f'Test accuracy: {test_acc:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1-score: {f1:.4f}')

    cm = confusion_matrix(all_labels, all_preds)
    classes = music_dataset.genres  

    plot_confusion_matrix(cm, classes, normalize=True, title='Matriz de Confusão Normalizada')
