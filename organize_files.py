import os
import shutil
import random

def split_data(source_dir, target_dir, train_ratio, random_seed=42):
    # Create directories for train, and test
    train_dir = os.path.join(target_dir, 'train')
    test_dir = os.path.join(target_dir, 'test')
    
    os.makedirs(train_dir, exist_ok=True) # exist ok is to avoid error if the directory already exists
    os.makedirs(test_dir, exist_ok=True)

    # Set the random seed
    random.seed(random_seed)

    # Iterate through each class directory in the source directory
    for class_dir in os.listdir(source_dir):
        class_path = os.path.join(source_dir, class_dir)

        if os.path.isdir(class_path): # check if it is a directory and not a file
            # Create corresponding class directories in train, and test
            os.makedirs(os.path.join(train_dir, class_dir), exist_ok=True)
            os.makedirs(os.path.join(test_dir, class_dir), exist_ok=True)

            # Get the list of files in the class directory 
            files = [f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))]
            
            # Shuffle the files
            random.shuffle(files)

            # Calculate the number of files for each split
            num_files = len(files)
            num_train = int(num_files * train_ratio)

            # Split the files into train, validate, and test
            train_files = files[:num_train]
            test_files = files[num_train:]
            

            # Copy the files to the corresponding class directories in train, validate, and test
            for file in train_files:
                shutil.copy(os.path.join(class_path, file), os.path.join(train_dir, class_dir, file))
            print("Copied files to train directory for class: ", class_dir, " with ", len(train_files), " files.")
            
            for file in test_files:
                shutil.copy(os.path.join(class_path, file), os.path.join(test_dir, class_dir, file))
            print("Copied files to test directory for class: ", class_dir, " with ", len(test_files), " files.")


if __name__ == "__main__":
    source_dir = 'DATASETS/merged_resized_pngs'
    target_dir = 'DATASETS/merged_resized_pngs_splited'
    train_ratio = 0.8
    

    split_data(source_dir, target_dir, train_ratio)
