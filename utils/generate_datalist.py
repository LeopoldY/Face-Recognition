import os
import numpy as np

def get_train_list(root, save_path):
    '''
    Args:
        - root: the root directory of the dataset

        the structure of the dataset should be:
        ```txt
        root
        ├── label1
        │   ├── image1
        │   ├── image2
        │   └── ...
        ├── label2
        │   ├── image1
        │   ├── image2
        │   └── ...
        ...
        ```
        - save_path: the path to save the list
        
        The list is in the format of:
        (whether the label is digit or not, it is labeled by the order of the directory)
        ```txt
        label1 image_path1
        label2 image_path2
        ...
        ```
    '''
    with open(save_path, 'w') as fd:
        class_num = 0
        for i in os.listdir(root):
            for j in os.listdir(os.path.join(root, i)):
                img_path = i + '/' + j
                fd.write(str(class_num) + ' ' + img_path + '\n')
            class_num += 1

def get_paired_eval_list(root, save_path, pair_num=6000, pos_neg_ratio=0.5):
    '''
    Args:
        - root: the root directory of the dataset

        the structure of the dataset should be:
        ```txt
        root
        ├── label1
        │   ├── image1
        │   ├── image2
        │   └── ...
        ├── label2
        │   ├── image1
        │   ├── image2
        │   └── ...
        ...
        ```
        - save_path: the path to save the list
        - pair_num: the number of pairs
        - pos_neg_ratio: the ratio of positive pairs in all pairs
    '''
    img_list = []
    img_dict = {}
    for i in os.listdir(root):
        for j in os.listdir(os.path.join(root, i)):
            img_list.append(i + '/' + j)
            if i not in img_dict:
                img_dict[i] = []
            img_dict[i].append(i + '/' + j)

    pos_sample_num = int(pair_num * pos_neg_ratio)
    neg_sample_num = pair_num - pos_sample_num

    pos_sample = []
    neg_sample = []

    # Pre-calculate the labels with at least 2 images
    valid_labels = [label for label in img_dict.keys() if len(img_dict[label]) >= 2]

    for i in range(neg_sample_num):
        img1, img2 = np.random.choice(img_list, 2, replace=False)
        if img1.split('/')[0] == img2.split('/')[0]:
            i -= 1
            continue
        neg_sample.append((img1, img2, 0))

    for i in range(pos_sample_num):
        label = np.random.choice(valid_labels)
        img1, img2 = np.random.choice(img_dict[label], 2, replace=False)
        pos_sample.append((img1, img2, 1))

    sample = pos_sample + neg_sample
    np.random.shuffle(sample)
    with open(save_path, 'w') as fd:
        for i in sample:
            fd.write(i[0] + ' ' + i[1] + ' ' + str(i[2]) + '\n')

if __name__ == '__main__':
    root = r'C:\Users\yangc\Developer\data\lfw-aligned'
    save_path = 'data/lfw-aligned.txt'
    get_paired_eval_list(root, save_path, pair_num=10000, pos_neg_ratio=0.5)