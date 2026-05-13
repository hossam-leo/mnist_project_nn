import matplotlib.pyplot as plt
import json
import glob
import os

def plot_experiments():
    result_files = glob.glob('results/*_results.json')
    
    plt.figure(figsize=(12, 5))
    
    # Plot Loss
    plt.subplot(1, 2, 1)
    for file in result_files:
        with open(file, 'r') as f:
            data = json.load(f)
        plt.plot(data['history']['loss'], label=data['name'])
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot Accuracy
    plt.subplot(1, 2, 2)
    for file in result_files:
        with open(file, 'r') as f:
            data = json.load(f)
        plt.plot(data['history']['accuracy'], label=data['name'])
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('results/experiment_comparison.png')
    plt.close()

def create_summary_table():
    result_files = glob.glob('results/*_results.json')
    summary = []
    
    for file in result_files:
        with open(file, 'r') as f:
            data = json.load(f)
        summary.append({
            'Experiment': data['name'],
            'Test Loss': f"{data['test_results']['loss']:.4f}",
            'Test Accuracy': f"{data['test_results']['accuracy']:.2f}%",
            'Test MSE': f"{data['test_results']['mse']:.4f}",
            'Activation': data['params']['activation'],
            'Learning Rate': data['params']['lr']
        })
    
    with open('results/summary_table.json', 'w') as f:
        json.dump(summary, f, indent=4)

if __name__ == "__main__":
    plot_experiments()
    create_summary_table()
