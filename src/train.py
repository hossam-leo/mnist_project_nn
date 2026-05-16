import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import json
import os
from model import MNIST_MLP

def load_data(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

def train(model, train_loader, criterion, optimizer, epochs=2):
    model.train()
    history = {'loss': [], 'accuracy': []}
    
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        history['loss'].append(epoch_loss)
        history['accuracy'].append(epoch_acc)
        print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')
        
    return history

def evaluate(model, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    mse_criterion = nn.MSELoss()
    total_mse = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += criterion(output, target).item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            # For MSE, we need one-hot encoded targets
            target_one_hot = torch.zeros(target.size(0), 10).scatter_(1, target.view(-1, 1), 1)
            output_softmax = torch.softmax(output, dim=1)
            total_mse += mse_criterion(output_softmax, target_one_hot).item()

    avg_loss = test_loss / len(test_loader)
    accuracy = 100. * correct / total
    avg_mse = total_mse / len(test_loader)
    
    return avg_loss, accuracy, avg_mse

def run_experiment(name, hidden_size=128, lr=0.01, activation='relu', epochs=5):
    print(f"\nRunning Experiment: {name}")
    train_loader, test_loader = load_data()
    model = MNIST_MLP(hidden_size=hidden_size, activation=activation)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    history = train(model, train_loader, criterion, optimizer, epochs=epochs)
    final_loss, final_acc, final_mse = evaluate(model, test_loader, criterion)
    
    print(f"Test Results - Loss: {final_loss:.4f}, Accuracy: {final_acc:.2f}%, MSE: {final_mse:.4f}")
    
    results = {
        'name': name,
        'params': {'hidden_size': hidden_size, 'lr': lr, 'activation': activation, 'epochs': epochs},
        'history': history,
        'test_results': {'loss': final_loss, 'accuracy': final_acc, 'mse': final_mse}
    }
    
    with open(f'results/{name}_results.json', 'w') as f:
        json.dump(results, f)
    
    return results

if __name__ == "__main__":
    os.makedirs('results', exist_ok=True)
    # Experiment 1: Baseline
    run_experiment("Baseline", hidden_size=128, lr=0.001, activation='relu')
    # Experiment 2: Higher Learning Rate
    run_experiment("High_LR", hidden_size=128, lr=0.01, activation='relu')
    # Experiment 3: Different Activation (Sigmoid)
    run_experiment("Sigmoid_Activation", hidden_size=128, lr=0.001, activation='sigmoid')
