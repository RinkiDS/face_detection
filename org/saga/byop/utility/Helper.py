import os

def get_image_files_from_dir(self, directory_path):
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        if os.path.isfile(file_path):
            print(f'Processing file: {filename}')

def load_model(model_path):
    model = load_model(model_path)
    print(model)