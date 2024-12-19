import pygad
import numpy as np
import torch
from sklearn.metrics import classification_report
from data_utils import split_dataset, MusicDataset, get_dataloaders
from model_utils import train_from_hf, plot_confusion_matrix
from transformers import AutoImageProcessor, AutoModelForImageClassification, AutoConfig
import os
import json
import argparse
from torchvision import transforms

def fitness_function(ga_instance, solution, solution_idx):
    """Fitness function to evaluate `val_loss`."""
    model_idx, lr, weight_decay, batch_size = solution
    model_name = LISTA_MODELOS[int(model_idx)]
    print(f"[Checkpoint] Using model {model_name} for solution {solution_idx} with batch_size={int(batch_size)}")

    feature_extractor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForImageClassification.from_pretrained(model_name)
    model = freeze_model_layers(model)  # Congela as camadas do modelo

    # Define transformations
    model_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=15),
        transforms.ToTensor(),
        transforms.Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)
    ])
    music_dataset.transform = model_transforms

    print("[Checkpoint] Loading training configuration")
    # Load training configuration
    with open(args.cfg, "r") as f:
        cfg = json.load(f)

    # Device setup
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(f"[Checkpoint] Evaluating solution {solution_idx}")
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.CrossEntropyLoss().to(device)

    # Create data loaders with dynamic batch size
    train_loader, val_loader, test_loader = get_dataloaders(music_dataset, batch_size=int(batch_size))

    best_val_loss = float('inf')

    for epoch in range(cfg['num_epochs']):
        print(f"[Checkpoint] Solution {solution_idx} - Epoch {epoch+1}/{cfg['num_epochs']} - Training")
        # Training loop
        model.train()
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.logits, labels)
            loss.backward()
            optimizer.step()

        # Validation loop
        print(f"[Checkpoint] Solution {solution_idx} - Epoch {epoch+1}/{cfg['num_epochs']} - Validation")
        val_loss = 0.0
        num_val_batches = len(val_loader)

        model.eval()
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs.logits, labels)
                val_loss += loss.item()

        val_loss /= num_val_batches

        # Update best validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"[Checkpoint] Solution {solution_idx} - New best val_loss: {best_val_loss:.4f}")

    return -best_val_loss  # Minimize loss (negative for PyGAD, which maximizes fitness)

LISTA_MODELOS = ['google/efficientnet-b0', 'microsoft/resnet-50', 'google/efficientnet-b1']

def callback_generation(ga_instance):
    """Callback to monitor the optimization progress."""
    print(f"[Checkpoint] Generation {ga_instance.generations_completed}: Best fitness = {ga_instance.best_solution()[1]}")

if __name__ == "__main__":
    print("[Checkpoint] Parsing arguments and loading data")
    # Parse arguments
    args = argparse.ArgumentParser()
    args.add_argument("--data_path", type=str, default="genres")
    args.add_argument("--model_name", type=str, default="google/efficientnet-b0")
    args.add_argument("--cfg", type=str, default="./cfg/training_cfg.json")
    args = args.parse_args()

    # Load dataset
    music_dataset = MusicDataset(os.path.join('data', args.data_path))
    train_dataset, val_dataset, test_dataset = split_dataset(music_dataset)

    print("[Checkpoint] Loading training configuration")
    # Load training configuration
    with open(args.cfg, "r") as f:
        cfg = json.load(f)

    print("[Checkpoint] Starting PyGAD optimization")
    # PyGAD setup
    num_generations = 5
    num_parents_mating = 2
    sol_per_pop = 4
    num_genes = 4  # model_idx, lr, weight_decay, and batch_size

    gene_space = [
        {"low": 0, "high": len(LISTA_MODELOS)},  # Índices dos modelos
        {"low": 1e-5, "high": 1e-2},                # Learning rate range
        {"low": 1e-6, "high": 1e-2},                # Weight decay range
        {"low": 16, "high": 32}                     # Batch size range (integer)
    ]

    ga_instance = pygad.GA(
        num_generations=num_generations,
        num_parents_mating=num_parents_mating,
        fitness_func=fitness_function,
        sol_per_pop=sol_per_pop,
        num_genes=num_genes,
        gene_space=gene_space,
        on_generation=callback_generation
    )

    ga_instance.run()

    print("[Checkpoint] Retrieving the best solution")
    # Retrieve the best solution
    best_solution, best_solution_fitness, _ = ga_instance.best_solution()
    best_model_idx, best_lr, best_weight_decay, best_batch_size = best_solution
    best_model_name = LISTA_MODELOS[int(best_model_idx)]
    print(f"Best solution: model={best_model_name}, lr={best_lr}, weight_decay={best_weight_decay}, batch_size={int(best_batch_size)}")
    print(f"Best fitness: {best_solution_fitness}")

    print("[Checkpoint] Training the final model with optimized hyperparameters")
    # Train the final model with the optimized hyperparameters
    feature_extractor = AutoImageProcessor.from_pretrained(best_model_name)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = AutoModelForImageClassification.from_pretrained(best_model_name).to(device)
    model = freeze_model_layers(model)  # Congela as camadas do modelo
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=best_lr, weight_decay=best_weight_decay)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    train_loader, val_loader, test_loader = get_dataloaders(music_dataset, batch_size=int(best_batch_size))
    train_from_hf(model, feature_extractor, train_loader, val_loader, optimizer, criterion, device=device, **cfg)

    print("[Checkpoint] Evaluating on the test set")
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs.logits, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    # Generate classification report
    classes = music_dataset.genres
    print("[Checkpoint] Generating classification report")
    print(classification_report(y_true, y_pred, target_names=music_dataset.classes))

    # Optionally, plot confusion matrix
    plot_confusion_matrix(y_true, y_pred, classes=music_dataset.classes)
