import numpy as np
import torch
from sklearn.metrics import classification_report
from data_utils import split_dataset, MusicDataset, get_dataloaders
from model_utils import train_from_hf, plot_confusion_matrix
from transformers import AutoImageProcessor, AutoModelForImageClassification
import os
import json
import argparse
import time
from torchvision import transforms
from functools import partial
from deap import base, creator, tools, algorithms
import skfuzzy as fuzz
from skfuzzy import control as ctrl

def freeze_model_layers(model):
    """Freeze all layers except the last one."""
    for param in model.parameters():
        param.requires_grad = False
    
    for param in model.classifier.parameters():
        param.requires_grad = True
    
    return model

def evaluate_model(model_idx, lr, weight_decay, batch_size, music_dataset, cfg, args):
    """Função de avaliação para um conjunto de hiperparâmetros."""
    model_name = LISTA_MODELOS[int(model_idx)]
    print(f"[Checkpoint] Avaliando modelo {model_name} com batch_size={int(batch_size)}, lr={lr}, weight_decay={weight_decay}")

    feature_extractor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForImageClassification.from_pretrained(model_name)
    model = freeze_model_layers(model)

    model_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=15),
        transforms.ToTensor(),
        transforms.Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)
    ])
    music_dataset.transform = model_transforms

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), 
                                lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.CrossEntropyLoss().to(device)

    train_loader, val_loader, test_loader = get_dataloaders(music_dataset, batch_size=int(batch_size))

    best_val_loss = float('inf')
    best_val_acc = 0.0
    best_val_f1 = 0.0
    inference_time = 0.0

    for epoch in range(cfg['num_epochs']):
        print(f"[Checkpoint] Época {epoch+1}/{cfg['num_epochs']} - Treinamento")
        
        model.train()
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.logits, labels)
            loss.backward()
            optimizer.step()

        print(f"[Checkpoint] Época {epoch+1}/{cfg['num_epochs']} - Validação")
        val_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        
        model.eval()
        start_time = time.time()
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs.logits, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.logits, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        inference_time = time.time() - start_time
        val_loss /= len(val_loader)
        val_acc = correct / total
        
        from sklearn.metrics import f1_score
        val_f1 = f1_score(all_labels, all_preds, average='macro')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
        if val_acc > best_val_acc:
            best_val_acc = val_acc
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            
        print(f"[Checkpoint] Época {epoch+1} - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")

    return best_val_loss, best_val_acc, best_val_f1, inference_time

def setup_fuzzy_system():
    """Configurar o sistema de inferência fuzzy."""
    
    val_loss = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'val_loss')
    accuracy = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'accuracy')
    f1_score = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'f1_score')
    inf_time = ctrl.Antecedent(np.arange(0, 10.1, 0.1), 'inference_time')
    
    model_quality = ctrl.Consequent(np.arange(0, 101, 1), 'model_quality')
    
    val_loss['baixo'] = fuzz.trimf(val_loss.universe, [0, 0, 0.3])
    val_loss['médio'] = fuzz.trimf(val_loss.universe, [0.2, 0.4, 0.6])
    val_loss['alto'] = fuzz.trimf(val_loss.universe, [0.5, 1, 1])
    
    accuracy['baixa'] = fuzz.trimf(accuracy.universe, [0, 0, 0.6])
    accuracy['média'] = fuzz.trimf(accuracy.universe, [0.5, 0.7, 0.9])
    accuracy['alta'] = fuzz.trimf(accuracy.universe, [0.8, 1, 1])
    
    f1_score['baixo'] = fuzz.trimf(f1_score.universe, [0, 0, 0.6])
    f1_score['médio'] = fuzz.trimf(f1_score.universe, [0.5, 0.7, 0.9])
    f1_score['alto'] = fuzz.trimf(f1_score.universe, [0.8, 1, 1])
    
    inf_time['rápido'] = fuzz.trimf(inf_time.universe, [0, 0, 2])
    inf_time['médio'] = fuzz.trimf(inf_time.universe, [1.5, 3, 5])
    inf_time['lento'] = fuzz.trimf(inf_time.universe, [4, 10, 10])
    
    model_quality['ruim'] = fuzz.trimf(model_quality.universe, [0, 0, 40])
    model_quality['médio'] = fuzz.trimf(model_quality.universe, [30, 50, 70])
    model_quality['bom'] = fuzz.trimf(model_quality.universe, [60, 80, 90])
    model_quality['excelente'] = fuzz.trimf(model_quality.universe, [80, 100, 100])
    
    rule1 = ctrl.Rule(val_loss['baixo'] & accuracy['alta'] & f1_score['alto'] & inf_time['rápido'], model_quality['excelente'])
    rule2 = ctrl.Rule(val_loss['baixo'] & accuracy['alta'] & f1_score['alto'] & inf_time['médio'], model_quality['bom'])
    rule3 = ctrl.Rule(val_loss['baixo'] & accuracy['alta'] & f1_score['médio'], model_quality['bom'])
    rule4 = ctrl.Rule(val_loss['médio'] & accuracy['média'], model_quality['médio'])
    rule5 = ctrl.Rule(val_loss['alto'] | accuracy['baixa'] | f1_score['baixo'], model_quality['ruim'])
    rule6 = ctrl.Rule(inf_time['lento'] & (accuracy['baixa'] | f1_score['baixo']), model_quality['ruim'])
    rule7 = ctrl.Rule(val_loss['médio'] & accuracy['alta'] & f1_score['alto'], model_quality['bom'])
    rule8 = ctrl.Rule(val_loss['baixo'] & accuracy['média'] & f1_score['médio'], model_quality['médio'])
    rule9 = ctrl.Rule(inf_time['lento'] & accuracy['alta'] & f1_score['alto'], model_quality['médio'])
    
    system = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9])
    
    return ctrl.ControlSystemSimulation(system)

def fuzzy_evaluate(metrics):
    """Avalia as métricas usando lógica fuzzy para determinar a qualidade do modelo."""
    val_loss, accuracy, f1, inference_time = metrics
    
    norm_val_loss = min(val_loss, 1.0)
    
    norm_inf_time = min(inference_time, 10.0)
    
    fuzzy_system = setup_fuzzy_system()
    
    fuzzy_system.input['val_loss'] = norm_val_loss
    fuzzy_system.input['accuracy'] = accuracy
    fuzzy_system.input['f1_score'] = f1
    fuzzy_system.input['inference_time'] = norm_inf_time
    
    try:
        fuzzy_system.compute()
        quality_score = fuzzy_system.output['model_quality']
        print(f"[Fuzzy] Val Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}, F1: {f1:.4f}, Inference Time: {inference_time:.2f}s")
        print(f"[Fuzzy] Model Quality Score: {quality_score:.2f}/100")
        return quality_score
    except:
        print("[Fuzzy] Erro ao computar a qualidade do modelo - usando avaliação padrão")
        return (accuracy * 40 + f1 * 40 - norm_val_loss * 10 - (norm_inf_time/10) * 10)

def fitness_function(individual, music_dataset, cfg, args):
    """Função de fitness usando o avaliador fuzzy."""
    model_idx, lr, weight_decay, batch_size = individual
    
    model_idx = int(model_idx)
    batch_size = int(batch_size)
    
    metrics = evaluate_model(model_idx, lr, weight_decay, batch_size, 
                            music_dataset, cfg, args)
    
    quality_score = fuzzy_evaluate(metrics)
    
    return quality_score,  

LISTA_MODELOS = ['google/efficientnet-b0', 'microsoft/resnet-50', 'google/efficientnet-b1']

def main():
    print("[Checkpoint] Analisando argumentos e carregando dados")
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="genres")
    parser.add_argument("--model_name", type=str, default="google/efficientnet-b0")
    parser.add_argument("--cfg", type=str, default="./cfg/training_cfg.json")
    args = parser.parse_args()

    music_dataset = MusicDataset(os.path.join('data', args.data_path))
    train_dataset, val_dataset, test_dataset = split_dataset(music_dataset)

    print("[Checkpoint] Carregando configuração de treinamento")

    with open(args.cfg, "r") as f:
        cfg = json.load(f)

    print("[Checkpoint] Configurando algoritmo genético com DEAP")
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    
    toolbox.register("attr_model", np.random.randint, 0, len(LISTA_MODELOS))
    toolbox.register("attr_lr", np.random.uniform, 1e-5, 1e-2)
    toolbox.register("attr_wd", np.random.uniform, 1e-6, 1e-2)
    toolbox.register("attr_batch", np.random.randint, 16, 33)  # [16, 32]
    
    toolbox.register("individual", tools.initCycle, creator.Individual,
                    (toolbox.attr_model, toolbox.attr_lr, toolbox.attr_wd, toolbox.attr_batch), n=1)
    
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    toolbox.register("evaluate", fitness_function, music_dataset=music_dataset, cfg=cfg, args=args)
    
    toolbox.register("mate", tools.cxBlend, alpha=0.5)  # Crossover
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=[0.5, 0.001, 0.001, 2], indpb=0.2)  # Mutação
    toolbox.register("select", tools.selTournament, tournsize=3)  # Seleção
    
    pop_size = 10
    n_gen = 10
    
    population = toolbox.population(n=pop_size)
    
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)
    
    print("[Checkpoint] Iniciando otimização com algoritmo genético")
    population, logbook = algorithms.eaSimple(population, toolbox, 
                                            cxpb=0.7,  # Probabilidade de crossover
                                            mutpb=0.3,  # Probabilidade de mutação
                                            ngen=n_gen, 
                                            stats=stats,
                                            verbose=True)
    
    best_ind = tools.selBest(population, k=1)[0]
    best_model_idx, best_lr, best_weight_decay, best_batch_size = best_ind
    best_model_name = LISTA_MODELOS[int(best_model_idx)]
    
    print("\n[Checkpoint] Melhores hiperparâmetros encontrados:")
    print(f"Modelo: {best_model_name}")
    print(f"Learning rate: {best_lr}")
    print(f"Weight decay: {best_weight_decay}")
    print(f"Batch size: {int(best_batch_size)}")
    print(f"Fitness (qualidade fuzzy): {best_ind.fitness.values[0]}")
    
    print("\n[Checkpoint] Treinando o modelo final com os hiperparâmetros otimizados")
    feature_extractor = AutoImageProcessor.from_pretrained(best_model_name)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = AutoModelForImageClassification.from_pretrained(best_model_name).to(device)
    model = freeze_model_layers(model)
    
    final_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=15),
        transforms.ToTensor(),
        transforms.Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)
    ])
    music_dataset.transform = final_transforms
    
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), 
                                lr=best_lr, weight_decay=best_weight_decay)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    
    train_loader, val_loader, test_loader = get_dataloaders(music_dataset, 
                                                          batch_size=int(best_batch_size))
    
    train_from_hf(model, feature_extractor, train_loader, val_loader, 
                optimizer, criterion, device=device, **cfg)
    
    print("[Checkpoint] Avaliando no conjunto de teste")
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
    
    print("[Checkpoint] Gerando relatório de classificação")
    print(classification_report(y_true, y_pred, target_names=music_dataset.classes))
    
    plot_confusion_matrix(y_true, y_pred, classes=music_dataset.classes)

if __name__ == "__main__":
    main()