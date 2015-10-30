__author__ = 'shawn'
import urllib, os, time
import zipfile
from progressbar import *

widgets = ['Test: ', Percentage(), ' ', Bar(marker=RotatingMarker()), ' ', ETA(), ' ', FileTransferSpeed()]
pbar = ProgressBar(widgets=widgets)

def dl_progress(count, blockSize, totalSize):
    if pbar.maxval is None:
        pbar.maxval = totalSize
        pbar.start()
    pbar.update(min(count*blockSize, totalSize))

def preprocessing():

    dataset_url = 'http://files.grouplens.org/datasets/movielens/ml-latest-small.zip'
    dataset_path = os.path.join('datasets', 'ml-latest-small.zip')
    # dataset_path = "datasets/ml-latest-small.zip"
    urllib.urlretrieve (dataset_url, dataset_path, reporthook=dl_progress)
    pbar.finish()

    with zipfile.ZipFile(dataset_path, "r") as z:
        z.extractall("datasets")

    pass


def simple_bar():
    bar = ProgressBar()
    bar.maxval = 20
    bar.start()
    for i in range(21):
        bar.update(i)
        time.sleep(0.1)

if __name__ == "__main__":
    print "preprocessing"
    # simple_bar()
    preprocessing()