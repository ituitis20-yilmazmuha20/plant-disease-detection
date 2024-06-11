from trainer_with_editted_classifier import TrainerWithEdittedClassifier
from base_trainer import Trainer
from custom_model import MobilnetWEdittedClassifier

import torchvision.models as models
from torch.optim.lr_scheduler import StepLR
import torch.nn as nn
import torch.optim as optim
import torch
import os

def train_base_model():
    BASE_PATH = "./"
    DATA_DIR = 'DATASETS/merged_resized_pngs_splited'
    NUM_CLASSES = len(os.listdir(os.path.join(BASE_PATH, f"{DATA_DIR}/train")))
    NUM_EPOCHS = 30
    BATCH_SIZE = 32
    SEED_VALUE = 42

    LEARNING_RATE = 0.005
    MOMENTUM = 0.9
    WEIGHT_DECAY = 0.0005

    STEP_SIZE = 10
    GAMMA = 0.1

    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    model.classifier[1] = nn.Linear(model.last_channel, NUM_CLASSES) # for mobilenet_v2

    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()
    scheduler = StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    trainer = Trainer(
        model=model,
        data_dir=DATA_DIR,
        num_classes=NUM_CLASSES,
        num_epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        seed_value=SEED_VALUE,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        device=device,
        base_path=BASE_PATH
    )
    
    trainer.run()

def train_model_with_editted_classifier():
    BASE_PATH = "./"
    DATA_DIR = 'DATASETS/merged_resized_pngs_splited_augmented'
    NUM_CLASSES = len(os.listdir(os.path.join(BASE_PATH, f"{DATA_DIR}/train")))
    NUM_EPOCHS = 30
    BATCH_SIZE = 32
    SEED_VALUE = 42

    LEARNING_RATE = 0.005
    MOMENTUM = 0.9
    WEIGHT_DECAY = 0.0005

    STEP_SIZE = 10
    GAMMA = 0.1

    num_plant_types = 15  # One-hot encoded plant type vector size
    model_with_editted_classifier = MobilnetWEdittedClassifier(num_classes=NUM_CLASSES, num_plant_types=num_plant_types)

    optimizer = optim.SGD(model_with_editted_classifier.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()
    scheduler = StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    trainer_with_editted_classifier = TrainerWithEdittedClassifier(
        model=model_with_editted_classifier,
        data_dir=DATA_DIR,
        num_classes=NUM_CLASSES,
        num_epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        seed_value=SEED_VALUE,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        device=device,
        base_path=BASE_PATH
    )
    
    trainer_with_editted_classifier.run()


if __name__ == "__main__":
    print("Training base model...")
    train_base_model()
    #print("Training model with edited classifier...")
    #train_model_with_editted_classifier()
    