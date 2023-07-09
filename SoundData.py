import os
import numpy as np
import torch
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader

class SoundDataParser:
    def __init__(self, DATA_PATH):
        self.DATA_PATH = DATA_PATH
        self.LABEL_PATH = os.path.join(self.DATA_PATH, "libriphone")
        self.TRAIN_PATH = os.path.join(self.DATA_PATH, "libriphone", "feat", "train")
        self.TEST_PATH = os.path.join(self.DATA_PATH, "libriphone", "feat", "test")

        self.train_dict = {}
        self.test_dict = {}

        self.parse_split_labels()
        self.parse_pt_directory()

    def parse_split_labels(self):
        # Parse train_split.txt and test_split.txt into dictionaries of id
        split_names = ["train", "test"]
        for split_name in split_names:
            with open(os.path.join(self.LABEL_PATH, f"{split_name}_split.txt"), 'r') as file:
                lines = file.readlines()
                for line in lines:
                    id = line.strip("\n")
                    if split_name == "train":
                        self.train_dict[id] = {}
                    elif split_name == "test":
                        self.test_dict[id] = {}
        # Parse train_label.txt
        with open(os.path.join(self.LABEL_PATH, "train_labels.txt"), 'r') as file:
            lines = file.readlines()
            for line in lines:
                line = line.strip("\n")
                text_list = line.split(" ")
                self.train_dict[text_list[0]] = {}
                self.train_dict[text_list[0]]["labels"] = np.array(text_list)[1:].astype(int)

    def parse_pt_directory(self):
        split_names = ["train", "test"]
        for split_name in split_names:
            split_path = None
            target_dict = None
            if split_name == 'train':
                split_path = self.TRAIN_PATH
                target_dict = self.train_dict
            elif split_name == 'test':
                split_path = self.TEST_PATH
                target_dict = self.test_dict

            pbar = tqdm(os.listdir(split_path))
            for filename in pbar:
                pbar.set_postfix({"file": f"{split_name} {filename}"})
                id = os.path.splitext(filename)[0]
                vectors = torch.load(os.path.join(split_path, filename)).numpy()
                target_dict[id]["vectors"] = vectors

class SoundDataset(Dataset):
    def __init__(self, sound_dict, window=5):
        # total frames: window*2 + 1
        self.x = []
        self.y = []

        dict_pbar = tqdm(sound_dict)
        for id in dict_pbar:
            # vectors to self.x
            T = sound_dict[id]["vectors"].shape[0] # (T, 39)
            for i in range(T):
                padded_vector = None
                start = i - window
                end = i + window+1
                front_pad = None
                tail_pad = None
                # Consider padding needed for front and tail
                if start < 0:
                    front_pad = np.zeros((-1*start, 39))
                    padded_vector = front_pad
                    start = 0
                if end > T:
                    tail_pad = np.zeros((end-T, 39))
                    end = T
                # Decide where to put middle
                if padded_vector is not None:
                    padded_vector = np.concatenate((padded_vector, sound_dict[id]["vectors"][start:end, :]))
                else:
                    padded_vector = sound_dict[id]["vectors"][start:end, :]
                # Concat tail if needed
                if tail_pad is not None:
                    padded_vector = np.concatenate((padded_vector, tail_pad))

                assert(padded_vector.shape==(2*window+1, 39))
                self.x.append(torch.FloatTensor(padded_vector))
            # labels to self.y
            if "labels" in sound_dict[id]:
                self.y.append(sound_dict[id]["labels"])
            else:
                self.y = None

        if self.y is not None:
            self.y = torch.tensor(np.concatenate(self.y, axis=0), dtype=torch.int64).view(-1, )
            # self.y = torch.nn.functional.one_hot(self.y).to(torch.float32)

    def __len__(self):
        if self.y is not None:
            assert len(self.x) == len(self.y)
        return len(self.x)

    def __getitem__(self, item):
        if self.y is not None:
            return self.x[item], self.y[item]
        else:
            return self.x[item]


if __name__ == "__main__":
    st = SoundDataParser(os.path.join(os.getcwd(), "data"))
    soundDataset = SoundDataset(st.train_dict)
    print("Running SoundData.py")