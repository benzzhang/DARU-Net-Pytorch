import os
import random


def randsplit():
    a = random.sample(range(0, 1000), 1000)
    return a[:200], a[200:400], a[400:600], a[600:800], a[800:]

def getFileDir(src_path, list):
    files = os.listdir(src_path)
    files.sort(key=lambda x: int(x[:-4]))
    for num, l in enumerate(list):
        print(l)
        with open(r'./KFold-test-{}.txt'.format(num), 'w') as f:
            for index in l:
                f.write(files[index] + '\n')

# Randomly divided into 5 fold data
# folder 'img' stored all images used for training
if __name__ == "__main__":
    a = []
    a1, a2, a3, a4, a5 = randsplit()
    a.append(a1)
    a.append(a2)
    a.append(a3)
    a.append(a4)
    a.append(a5)
    getFileDir(r'C:\Users\maxwell\Desktop\6-4\img', a)