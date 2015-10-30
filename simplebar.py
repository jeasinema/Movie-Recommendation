import progressbar
import time
pbar = progressbar.ProgressBar()
pbar.start()
pbar.maxval = 100
for i in range(101):
    pbar.update(i)
    time.sleep(0.01)
pbar.finish()
