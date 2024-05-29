from trainer import Trainer
import torchvision.models as models
from torch.optim.lr_scheduler import StepLR
import torch.nn as nn
import torch.optim as optim
import torch
import os


if __name__ == "__main__":
    BASE_PATH = "./"
    DATA_DIR = 'DATASETS/merged_resized_pngs_splited_augmented'
    #MODEL_PATH = 'models/GoogLeNet_20240324_220402/best_model.pth' # Path to the best model
    NUM_CLASSES = len(os.listdir(os.path.join(BASE_PATH, f"{DATA_DIR}/train")))
    NUM_EPOCHS = 30
    BATCH_SIZE = 32
    SEED_VALUE = 42

    # Configuration for optimizer
    LEARNING_RATE = 0.005
    MOMENTUM = 0.9
    WEIGHT_DECAY = 0.0005

    # Configurations for learning rate scheduler
    STEP_SIZE = 10
    GAMMA = 0.1

    # Load the model
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    model.classifier[1] = nn.Linear(model.last_channel, NUM_CLASSES) # for mobilenet_v2
    # model.fc = nn.Linear(1024, NUM_CLASSES)  for googlenet
    # model.load_state_dict(torch.load(MODEL_PATH)) # for fine-tuning
    
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
    