import os
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

REBUILD_DATA = False

class DosVSCats():
    IMGSIZE = 50
    Cats = '/Volumes/SanDisk_SSD/Files/dataset/pics/PetImages/Cat'
    Dogs = '/Volumes/SanDisk_SSD/Files/dataset/pics/PetImages/Dog'
    Lables = {Cats:0, Dogs:1}
    training_data = []

    catcount = 0
    dogcount = 0

    def make_training_data(self):
        for label in self.Lables:
            print(label)
            for i in tqdm(os.listdir(label)):
                if "jpg" in i:
                    try:
                        path = os.path.join(label,i)
                        img = cv2.imread(path, cv2.IMREAD_COLOR)
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        img = cv2.resize(img, (self.IMGSIZE, self.IMGSIZE))
                        img = np.resize(img, (3,50,50))
                        self.training_data.append([np.array(img), np.eye(2)[self.Lables[label]]])

                    except Exception as e:
                        pass

        np.random.shuffle(self.training_data)
        np.save('training_data_color.npy', self.training_data)
        print('Cats:', dosVSCats.catcount)
        print("dogs:", DosVSCats.dogcount)


if REBUILD_DATA:
    dosVSCats = DosVSCats()
    dosVSCats.make_training_data()

training_data = np.load("training_data_color.npy",  allow_pickle=True)
# plt.imshow(training_data[0][0])
# plt.show()
print(len(training_data))
