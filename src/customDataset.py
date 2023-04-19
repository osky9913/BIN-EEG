import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, data_folder, subjects_range, series_range, train_test_split_ratio, train=True):
        self.input_data = []
        self.output_data = []

        for subject in subjects_range:
            print(subject)
            for series in series_range:
                print(series)

                input_filename = f"subj{subject}_series{series}_data.csv"
                output_filename = f"subj{subject}_series{series}_events.csv"

                input_filepath = os.path.join(data_folder, input_filename)
                output_filepath = os.path.join(data_folder, output_filename)
                print(input_filepath)
                print(output_filepath)
                if not os.path.exists(input_filepath) or not os.path.exists(output_filepath):
                    continue

                input_df = pd.read_csv(input_filepath)
                output_df = pd.read_csv(output_filepath)

                # Drop the 'id' column
                input_df.drop('id', axis=1, inplace=True)
                output_df.drop('id', axis=1, inplace=True)

                input_data = input_df.to_numpy(np.float32)
                output_data = output_df.to_numpy(np.float32)

                self.input_data.append(input_data)
                self.output_data.append(output_data)

        self.input_data = np.vstack(self.input_data)
        self.output_data = np.vstack(self.output_data)
        print("All ", len(self.input_data))

        split_index = int(len(self.input_data) * train_test_split_ratio)
        if train:
            self.input_data = self.input_data[:split_index]
            self.output_data = self.output_data[:split_index]
            print("Train ", len(self.input_data))
        else:
            self.input_data = self.input_data[split_index:]
            self.output_data = self.output_data[split_index:]
            print("Test ", len(self.output_data))


    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, index):
        input_data = torch.tensor(self.input_data[index]).float().unsqueeze(0)
        output_data = torch.tensor(self.output_data[index]).float()
        #print(input_data,input_data.size())
        #print(output_data, output_data.size())
        return input_data, output_data

    def to(self, device):
        self.input_data = torch.from_numpy(self.input_data).to(device)
        self.output_data = torch.from_numpy(self.output_data).to(device)