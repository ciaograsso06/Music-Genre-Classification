# Music Genre Classification with Pretrained Model

This project is a Python-based implementation for classifying music genres using a pretrained image classification model from Hugging Face Transformers library. The pipeline processes spectrogram images of music and leverages `google/efficientnet-b0` to achieve accurate predictions.

## Features

- Uses pretrained models for transfer learning.
- Supports data augmentation.
- Implements early stopping for training.
- Saves the best model during training.
- Generates classification metrics including confusion matrix and precision/recall scores.

## Requirements

The following libraries are required to run the script:

```bash
pip install torch torchvision transformers tqdm sklearn matplotlib seaborn
```

## Script Overview

### Imports

```python
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
```

### NoneTransform

A utility class to handle cases where no transformation is required:

```python
class NoneTransform(object):
    def __call__(self, image):
        return image
```

### Main Script

#### Argument Parsing

Command-line arguments for specifying dataset path, model name, and training configuration:

```python
if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--data_path", type=str, default="genres")
    args.add_argument("--model_name", type=str, default="google/efficientnet-b0")
    args.add_argument("--cfg", type=str, default="./cfg/training_cfg.json")
    args.add_argument('--fake_rgb', action='store_true')
    args = args.parse_args()
```

#### Dataset Preparation

```python
music_dataset = MusicDataset(os.path.join('data', args.data_path))
```

#### Model Configuration

```python
config = AutoConfig.from_pretrained(args.model_name)
config.num_labels = music_dataset.genres.shape[0]

feature_extractor = AutoImageProcessor.from_pretrained(args.model_name)
model = AutoModelForImageClassification.from_pretrained(args.model_name)
```

#### Data Augmentation

```python
model_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees=15),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)) if args.fake_rgb else NoneTransform(),
    transforms.Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)
])
music_dataset.transform = model_transforms
```

#### Training and Evaluation

Training loop with early stopping:

```python
train_dataset, val_dataset, test_dataset = split_dataset(music_dataset)
train_loader, val_loader, test_loader = get_dataloaders(music_dataset)

with open(args.cfg, "r") as f:
    cfg = json.load(f)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
name = cfg['name']
lr = cfg['lr']
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=cfg['weight_decay'])
criterion = nn.CrossEntropyLoss().to(device)

train_from_hf(model, image_processor, train_loader, val_loader, optimizer, criterion, device=device, **cfg)
```

Saving the best model:

```python
best_val_loss = float('inf')
patience = 5
patience_counter = 0

for epoch in range(cfg['num_epochs']):
    # Training loop
    ...

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        model.save_pretrained(os.path.join('ckpt', f'{name}_best_model_true'))
        image_processor.save_pretrained(os.path.join('ckpt', f'{name}_best_model_true'))
        patience_counter = 0
    else:
        patience_counter += 1

    if patience_counter >= patience:
        print("Early stopping")
        break
```

#### Testing and Metrics

Evaluate the model on the test set:

```python
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
```

Generate confusion matrix:

```python
cm = confusion_matrix(all_labels, all_preds)
classes = music_dataset.genres

plot_confusion_matrix(cm, classes, normalize=True, title='Matriz de Confusão Normalizada')
```

## Running the Script

Execute the script using the following command:

```bash
python music_classification.py --data_path images_original
```

## License

This project is licensed under the MIT License. See the LICENSE file for details.

