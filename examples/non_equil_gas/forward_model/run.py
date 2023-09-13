# import numpy as np
import yaml
import torch
from solver import SOLVER


def configure_model(config_data, device):
    """Configure the model based on the config file"""


def load_config(config_file_path="./config.yml"):
    """Load the config file
    Input:
    config_file_path: path to the config file
    Output:
    config_data: dictionary containing the config data
    """
    with open(config_file_path, "r") as config_file:
        config_data = yaml.load(config_file, Loader=yaml.FullLoader)
    return config_data


def main():
    # Load the config file
    config_data = load_config()

    # Device definition
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

    # Model definition
    SOLVER(config_data, device)


if __name__ == "__main__":
    main()
