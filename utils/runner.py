import time
import torch

from tqdm import tqdm
from torch.utils.data import dataloader

from .dataset import FolderDataset


class Runner():
    def __init__(self, model, dataset: FolderDataset):
        self.model = model.eval().to("cuda:0")
        self.dataset = dataset
        self.dataloader = dataloader.DataLoader(
            dataset, 
            batch_size=1,
            shuffle=False
        ) # DO NOT change ANY options

    @torch.no_grad()
    def run(self):
        print('Detection model inferring ...')
        predictions = {}

        start_time = time.time()
        for name, img in tqdm(zip(self.dataset.get_img_name(), self.dataloader)):
            img = img.to('cuda:0')
            pred = self.model(img).detach().cpu().numpy().squeeze().tolist()
            predictions[name] = pred
        end_time = time.time()

        return {"predictions": predictions, "time": end_time - start_time}
