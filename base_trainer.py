import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torch.optim.lr_scheduler import StepLR
import os
import pandas as pd
import random 
import numpy as np
from datetime import datetime
import json
import logging
import yaml
from sklearn.metrics import precision_recall_fscore_support


#  We will use class structure for training process

class Trainer:
    def __init__(self, model, data_dir, num_classes, num_epochs, batch_size, seed_value, optimizer, criterion, device, base_path, scheduler=None):
        self.base_path = base_path
        self.data_dir = data_dir 
        self.num_classes = num_classes
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.seed_value = seed_value
        self.device = device
        self.set_seed(seed_value)
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_loader, self.test_loader = self.load_data()
        self.scheduler = scheduler if scheduler else StepLR(self.optimizer, step_size=10, gamma=0.1)
        self.best_test_loss = float('inf')
        self.model_dir = os.path.join(self.base_path, f"models/{self.model.__class__.__name__}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        self.logger = self.setup_logger()
        self.config = self.create_configuration_dict()  # configuration dictionary for model, optimizer, and scheduler
        
        
    def create_configuration_dict(self):
        config = {
            "model": self.model.__class__.__name__,
            "optimizer": {
                "type": self.optimizer.__class__.__name__,
                "learning_rate": self.optimizer.param_groups[0]["lr"],
                "momentum": self.optimizer.param_groups[0]["momentum"],
                "weight_decay": self.optimizer.param_groups[0]["weight_decay"]
            },
            "scheduler": {
                "type": self.scheduler.__class__.__name__,
                "step_size": self.scheduler.step_size,
                "gamma": self.scheduler.gamma
            },
            "num_epochs": self.num_epochs,
            "batch_size": self.batch_size
        }

        return config

    def setup_logger(self):
        logger = logging.getLogger('training_logger')
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # log to console
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # log to file
        os.makedirs(self.model_dir, exist_ok=True)
        file_handler = logging.FileHandler(os.path.join(self.model_dir, 'training.log'))
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        return logger
    
    def close_logger(self):
        for handler in self.logger.handlers:
            handler.close()
            self.logger.removeHandler(handler)

    def set_seed(self, seed_value):
        torch.manual_seed(seed_value) # Set for CPU
        torch.cuda.manual_seed_all(seed_value) # Set for GPU
        np.random.seed(seed_value) # Set for numpy
        random.seed(seed_value) # Set for python
        torch.backends.cudnn.deterministic = True # Set for cudnn
        torch.backends.cudnn.benchmark = False # Set for cudnn

    def load_data(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Normalize the images with ImageNet mean and std
        ])

        os.path.join(self.base_path, f'{self.data_dir}/train')

        train_dataset = ImageFolder(root=os.path.join(self.base_path, f'{self.data_dir}/train'), transform=transform)
        test_dataset = ImageFolder(root=os.path.join(self.base_path, f'{self.data_dir}/test'), transform=transform)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4) # Shuffle training set for every epoch
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4) # No need to shuffle test set

        return train_loader, test_loader
    

    def train_model(self):
        self.model.to(self.device)
        assert next(self.model.parameters()).is_cuda, "Check if model is loaded to correct device" # Check if model is loaded to correct device

        test_accuracies = []
        test_losses = []
        test_precisions = []
        test_recalls = []
        test_f1_scores = []

        train_losses = []    

        self.logger.info("TRAINING STARTED...")
        self.logger.info(f"Training on {self.device}")
        for epoch in range(self.num_epochs):
            self.logger.info(f"Epoch {epoch+1}/{self.num_epochs}, Training...")

            self.model.train()
            running_loss = 0.0
            for images, labels in self.train_loader:
                images, labels = images.to(self.device), labels.to(self.device) # Load data to device
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step() # Update weights and biases
                running_loss += loss.item()

            self.scheduler.step() # Update learning rate if step_size is reached

            # Save training results for each epoch
            train_loss = running_loss / len(self.train_loader) # Calculate average training loss for each epoch
            train_losses.append((epoch + 1, train_loss)) 
            self.logger.info(f"Epoch {epoch+1}/{self.num_epochs}, Training Loss: {train_loss:.4f}") # log training loss

            # Save test results for each epoch
            #self.logger.info(f"Epoch {epoch+1}/{self.num_epochs}, Evaluating model on test set...")
            test_accuracy, test_loss, precision, recall, f1_score = self.evaluate_model(self.test_loader)
            test_accuracies.append((epoch + 1, test_accuracy))
            test_losses.append((epoch + 1, test_loss))
            test_precisions.append((epoch + 1, precision))
            test_recalls.append((epoch + 1, recall))
            test_f1_scores.append((epoch + 1, f1_score))
            self.logger.info(f"Epoch {epoch+1}/{self.num_epochs}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%") # log test loss and accuracy

            # Checkpointing based on test loss
            if test_loss < self.best_test_loss:
                self.best_test_loss = test_loss
                model_path = os.path.join(self.model_dir, "best_model.pth")
                os.makedirs(self.model_dir, exist_ok=True)
                torch.save(self.model.state_dict(), model_path)
                self.logger.info(f"Saved new best model with test loss: {test_loss:.4f}")

            # Create DataFrame for training results
        train_results_df = pd.DataFrame({
            "Epoch": [epoch for epoch in range(1, self.num_epochs+1)],
            "Training Loss": [round(loss, 4) for _, loss in train_losses]
        })

        # Create DataFrame for test results
        test_results_df = pd.DataFrame({
            "Epoch": [epoch for epoch, _ in test_accuracies],
            "Test Accuracy": [round(accuracy, 4) for _, accuracy in test_accuracies],
            "Test Loss": [round(loss, 4) for _, loss in test_losses],
            "Test Precision": [round(precision, 4) for _, precision in test_precisions],
            "Test Recall": [round(recall, 4) for _, recall in test_recalls],
            "Test F1 Score": [round(f1_score, 4) for _, f1_score in test_f1_scores]
        })

        return train_results_df, test_results_df

    def evaluate_model(self, data_loader):
        self.model.eval()
        correct = 0
        total = 0
        total_loss = 0.0 # We will also calculate loss for test to use it for checkpointing (saving best model based on test loss)
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for images, labels in data_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels) # Calculate loss
                total_loss += loss.item() # Add loss to total_loss

                _, predicted = torch.max(outputs, 1)
                all_predictions.extend(predicted.cpu().numpy()) # Add predictions to all_predictions list cpu() is used to move data to cpu
                all_targets.extend(labels.cpu().numpy()) # Add targets to all_targets list

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        avg_loss = total_loss / len(data_loader) # Calculate average loss

        # Calculate confusion matrix
        precision, recall, f1_score, _ = precision_recall_fscore_support(all_targets, all_predictions, average='weighted', zero_division=1) # weighted since we have imbalanced dataset

        return accuracy, avg_loss, precision, recall, f1_score

    def run(self):
        train_results_df, test_results_df = self.train_model()

        # Save configuration to json file
        config_path = os.path.join(self.model_dir, "config.json")
        with open(config_path, "w") as config_file:
            json.dump(self.config, config_file, indent=4)

        # Save model the model
        model_path = os.path.join(self.model_dir, "model.pth")
        torch.save(self.model.state_dict(), model_path)

        # Save results to csv files
        train_results_df.to_csv(os.path.join(self.model_dir, "train_results.csv"), index=False)
        test_results_df.to_csv(os.path.join(self.model_dir, "test_results.csv"), index=False)

        self.logger.info(f"Model and results saved to {self.model_dir}")

        # close logger handlers so that we can reuse the logger for another training
        self.close_logger()

        
        
