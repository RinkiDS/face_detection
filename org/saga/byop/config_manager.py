import yaml
import os
# Global variable to store configuration settings
CONFIG = None

# Function to load configuration settings from the YAML file
def load_config(file_path):
    global CONFIG
    with open(file_path, 'r') as file:
        CONFIG = yaml.safe_load(file)

# Load configuration settings at the start of your application
cwd=os.getcwd()
load_config(os.path.join(cwd,'config.yaml'))

# Access configuration settings throughout your application
def get_model_path():
    return CONFIG['paths']['model_path']

def get_directory_path():
    return CONFIG['paths']['directory_path']

# Example usage
if __name__ == "__main__":

    print("App Name:", get_model_path())
    print("Logging Level:", get_directory_path())
