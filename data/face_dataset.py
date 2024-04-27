import argparse
import os

import cv2
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision import transforms as T
import random
from tqdm import tqdm

class FaceDataset(Dataset):
    def __init__(self, data_root, description, transform = None, shuffle=True, train=True):
        '''
        Args:
            - data_root: the root path of the dataset.
            - description: the path of the description file.
            
            the train description file should be in the format of:
            ```txt
            label1 PATH/TO/IMAGE1
            label2 PATH/TO/IMAGE2
            ...
            ```
            the evaluation description file should be in the format of:
            ```txt
            PATH/TO/IMAGE1 PATH/TO/IMAGE2 label1
            PATH/TO/IMAGE2 PATH/TO/IMAGE2 label2
            ...
            ```
            - transform: the transformation to apply to the images.
            - shuffle: whether to shuffle the dataset.
            - train: whether to load the dataset for training or evaluation.
        '''
        super(FaceDataset, self).__init__()
        self.root = data_root
        self.shuffle = shuffle
        self.train = train
        self.transform = transform
        self.class_num = 0
        self.labels = []
        
        if self.train:
            self.images_path = []
            self.img_dict = {}

            if self.transform is None:
                self.transform = T.Compose([
                    T.ToTensor(),
                    T.RandomHorizontalFlip(),
                    T.Resize((112, 112)),
                    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ])

            # read the description file and generate data list and data dictionary
            with open(description) as f:
                lines = f.readlines()
            for l in lines:
                words = l.rstrip('\n').split(' ')
                
                abs_path = os.path.join(self.root, words[1])
                self.images_path.append(abs_path)
                self.labels.append(int(words[0]))
                
                if int(words[0]) not in self.img_dict:
                    self.img_dict[int(words[0])] = []
                else:
                    self.img_dict[int(words[0])].append(abs_path)

            assert len(self.images_path) == len(self.labels)  
            self.class_num = len(self.img_dict)
            if self.shuffle:
                self.__shuffle()
            print(f"loading {len(self.images_path)} samples in {self.class_num} classes.")
        else:
            self.img_pairs = []

            if self.transform is None:
                self.transform = T.Compose([
                    T.ToTensor(),
                    T.Resize((112, 112)),
                    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ])
            
            with open(description) as f:
                lines = f.readlines()
            for l in lines:
                words = l.rstrip('\n').split(' ')
                img1 = os.path.join(self.root, words[0])
                img2 = os.path.join(self.root, words[1])
                self.img_pairs.append([img1, img2])
                self.labels.append(int(words[2]))
            assert len(self.img_pairs) == len(self.labels)
            print(f"loading {len(self.img_pairs)} pairs.")

    def get_class_num(self):
        return self.class_num
    
    def __shuffle(self):
        zipped = list(zip(self.images_path, self.labels))
        random.shuffle(zipped)
        self.images_path, self.labels = zip(*zipped)

    # def __get_pos_sample(self, label):
    #     size = len(self.img_dict[label])
    #     for i in range(self.__mini_batch):
    #         imgIndex = np.random.randint(size)
    #         img = cv2.imread(self.img_dict[label][imgIndex])
    #         if self.transform:
    #             img = self.transform(img).unsqueeze(0)
    #         if i == 0:
    #             imgList = img
    #         else:
    #             imgList = torch.cat((imgList, img), 0)
    #     labelList = torch.ones(self.__mini_batch) * label
    #     return imgList, labelList
    
    # def __get_neg_sample(self, pos_label):
    #     idx = np.random.randint(self.__len__() - 1)
    #     for i in range(self.__mini_batch):
    #         while self.labels[idx] == pos_label:
    #             idx = np.random.randint(self.__len__() - 1)

    #         img = cv2.imread(self.images_path[idx])
    #         if self.transform:
    #             img = self.transform(img).unsqueeze(0)
    #         if i == 0:
    #             imgList = img
    #         else:
    #             imgList = torch.cat((imgList, img), 0)
    #     idxList = torch.zeros(self.__mini_batch)
    #     return imgList, idxList

    def __getitem__(self, idx):
        # if idx >= len(self.labels):
        #     raise IndexError
        # img = cv2.imread(self.images_path[idx])
        # if self.transform:
        #     img = self.transform(img)

        # label = self.labels[idx]
        # if self.__mini_batch > 0:
        #     pos_img, pos_label = self._get_pos_sample(label)
        #     neg_img, neg_label = self._get_neg_sample(label)

        #     imgs = torch.cat((img, pos_img, neg_img), 0)
        #     labels = torch.cat((label, pos_label, neg_label), 0)
        
        if self.train:
            img = cv2.imread(self.images_path[idx])
            if self.transform:
                img = self.transform(img)
            label = self.labels[idx]

            return img, label
        else:
            img1 = cv2.imread(self.img_pairs[idx][0])
            img2 = cv2.imread(self.img_pairs[idx][1])
            if self.transform:
                img1 = self.transform(img1)
                img2 = self.transform(img2)
            label = self.labels[idx]
            return img1, img2, label

    def __len__(self):
        return len(self.labels)
    
# class FaceDataBatchSampler(FaceDataSampler):
#     def __init__(self, data: Dataset, batch_size: int, shuffle: bool) -> None:
#         super().__init__(data)
#         self.batch_size = batch_size
#         self.shuffle = shuffle

#         print(f"FaceDataBatchSampler: batch_size={batch_size}, shuffle={shuffle}")

#     def __iter__(self):
#         if self.shuffle:
#             self.order = np.random.permutation(len(self.data_source))
#         for i in range(0, len(self.data_source), self.batch_size):
#             yield self.order[i:i+self.batch_size]

#     def __len__(self) -> int:
#         return len(self.data_source) // self.batch_size
    

class FaceDataloader(DataLoader):
    def __init__(self, dataset: Dataset,
                batch_size=8,
                num_workers=0,
                pin_memory=True,
                shuffle=True):
        
        # self.neg_dataset = NegativeDataset(dataset_dir, mini_batch=batch_size)
        # self.neg_loader = DataLoader(dataset=self.neg_dataset,
        #                              batch_size=batch_size,
        #                              shuffle=shuffle)
        # self.shuffle = shuffle
        # self.mini_batch = batch_size
        # self.dataset = PositiveDataset(dataset_dir, mini_batch=batch_size)
        # super(FaceDataloader, self).__init__(dataset=self.dataset,
        #                                      batch_size=1,
        #                                      shuffle=shuffle)
        
        self.batch_size = batch_size
        self.dataset = dataset
        self.num_workers = num_workers



        super(FaceDataloader, self).__init__(dataset=dataset,
                                             batch_size=batch_size,
                                             num_workers=num_workers,
                                             pin_memory=pin_memory,
                                             shuffle=shuffle)

    def __iter__(self):
        base_batch = super(FaceDataloader, self).__iter__()
        for batch, labels in base_batch:
            # neg_batch, neg_lables = self.neg_loader.__iter__().__next__()
            # pos_batch = pos_batch.squeeze(0)
            # pos_lable = pos_lable.squeeze(0)
            # target = pos_batch[0].unsqueeze(0)
            # target_label = pos_lable[0].unsqueeze(0)
            # pos_batch = pos_batch[1:]
            # pos_lable = pos_lable[1:]
            # if self.shuffle:
            #     shuffle_musk = torch.randperm(self.mini_batch * 2)
            #     valid_batch = torch.cat((pos_batch, neg_batch), 0)
            #     valid_label = torch.cat((pos_lable, neg_lables), 0)
            #     valid_batch = valid_batch[shuffle_musk]
            #     valid_label = valid_label[shuffle_musk]
            # batch = torch.cat((target, valid_batch), 0)
            # labels = torch.cat((target_label, valid_label), 0)
            yield  batch, labels

    def __len__(self):
        return len(self.dataset) // self.batch_size


if __name__ == '__main__':
    cfg = yaml.safe_load(open("config.yml", "r"))
    args = argparse.Namespace()
    for key in cfg:
        setattr(args, key, cfg[key])

    FD = FaceDataset(data_root=args.dataset_root,
                    description=args.datalist,
                    shuffle=True)

    data = Dataset()

    FRD = FaceDataloader(FD,
                        batch_size=args.batch_size,
                        num_workers=0)
    
    train_bar = tqdm(FRD)
    for batch, label in train_bar:
        print(batch.shape, label.shape)
        break