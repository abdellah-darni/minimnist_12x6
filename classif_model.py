import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


class DigitDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class NN(nn.Module):

    def __init__(self):
        super(NN, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(72, 20),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(20, 10),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(10, 10)
        )

    def forward(self, x):
        return self.model(x)


def save_model(model, optimizer, epoch, train_loss, save_path):
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'train_loss': train_loss
    }

    torch.save(checkpoint, save_path)
    print(f"Model checkpoint saved to {save_path}")


def load_model(model, optimizer, load_path, device):
    try:
        checkpoint = torch.load(load_path, map_location=device)

        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        last_epoch = checkpoint['epoch']
        train_loss = checkpoint['train_loss']

        print(f"Loaded model checkpoint from epoch {last_epoch}")
        return last_epoch, train_loss

    except FileNotFoundError:
        print("No previous checkpoint found. Starting from scratch.")
        return 0, 0


def train_model(model, train_loader, criterion, optimizer, num_epochs, device, test_loader=None):
    train_losses = []
    train_accuracies = []
    test_accuracies = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        train_accuracy = 100 * correct / total

        train_losses.append(epoch_loss)
        train_accuracies.append(train_accuracy)

        if test_loader is not None:
            test_accuracy = test_model(model, test_loader, device)
            test_accuracies.append(test_accuracy)
            print(
                f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Test Accuracy: {test_accuracy:.2f}%')
        else:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%')

    return train_losses, train_accuracies, test_accuracies


def test_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy


def plot_metrics(train_losses, train_accuracies, test_accuracies):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train')
    if test_accuracies:
        plt.plot(test_accuracies, label='Test')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.tight_layout()
    plt.show()


def main():
    # Configuration
    RANDOM_SEED = 42                        #42
    BATCH_SIZE = 64                         #64
    NUM_EPOCHS = 500                        #500
    LEARNING_RATE = 0.00005                 #0.00005
    MODEL_SAVE_PATH = "./models/classification/classif_model.pth"
    CONTINUE_TRAINING = True

    # Set random seeds for reproducibility
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load and preprocess data
    print("Loading data...")
    try:
        # Load training data
        train_data = pd.read_csv('final_datasets/train_dataset.csv')
        feature_columns = [f'pixel_{i}' for i in range(72)]
        X_train = train_data[feature_columns].values
        y_train = train_data['label'].values

        # Load testining data
        test_data = pd.read_csv('final_datasets/test_dataset.csv')
        feature_columns = [f'pixel_{i}' for i in range(72)]
        X_test = test_data[feature_columns].values
        y_test = test_data['label'].values


    except Exception as e:
        print(f"Error loading data: {e}")
        return


    # Create datasets and dataloaders
    train_dataset = DigitDataset(X_train, y_train)
    test_dataset = DigitDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Initialize model and optimizer
    model = NN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    # Load previous checkpoint if it exists and CONTINUE_TRAINING is True
    start_epoch = 0
    if CONTINUE_TRAINING:
        start_epoch, prev_loss = load_model(model, optimizer, MODEL_SAVE_PATH, device)

    # Train model
    print("Starting training...")
    train_losses, train_accuracies, test_accuracies = train_model(
        model, train_loader, criterion, optimizer, NUM_EPOCHS - start_epoch, device, test_loader
    )

    # Plot training metrics
    plot_metrics(train_losses, train_accuracies, test_accuracies)

    # Final test accuracy
    final_test_accuracy = test_model(model, test_loader, device)
    print(f'\nFinal Test Accuracy: {final_test_accuracy:.2f}%')

    # Save model
    save_model(model, optimizer, NUM_EPOCHS, train_losses[-1], MODEL_SAVE_PATH)


if __name__ == "__main__":
    main()
