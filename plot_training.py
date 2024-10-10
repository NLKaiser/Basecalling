import matplotlib.pyplot as plt
import numpy as np

epoch = []
train_loss = []
val_loss = []
val_mean_accuracy = []
val_median_accuracy = []

with open("training.csv") as f:
    next(f)
    for line in f:
        l = line.split(",")
        epoch.append(int(l[0]))
        train_loss.append(float(l[1]))
        val_loss.append(float(l[2]))
        val_mean_accuracy.append(float(l[3]))
        val_median_accuracy.append(float(l[4]))

train_loss = np.nan_to_num(train_loss, nan=1000, posinf=1000, neginf=1000)
train_loss = np.clip(train_loss, a_min=None, a_max=1000)
val_loss = np.nan_to_num(val_loss, nan=1000, posinf=1000, neginf=1000)
val_loss = np.clip(val_loss, a_min=None, a_max=1000)

plt.plot(epoch, train_loss, label='Training Loss', color='blue', marker='.', markersize=6, linestyle='None')
plt.plot(epoch, val_loss, label='Validation Loss', color='red', marker='*', markersize=2, linestyle='None')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.xlim(0, len(epoch))
plt.ylim(0, max(max(train_loss), max(val_loss)))
plt.savefig('loss.png')
plt.show()
plt.clf()

plt.plot(epoch, val_mean_accuracy, label='Mean accuracy', color='orange', marker='o', markersize=6, linestyle='None')
plt.plot(epoch, val_median_accuracy, label='Median accuracy', color='grey', marker='+', markersize=4, linestyle='None')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.xlim(0, len(epoch))
plt.ylim(0, 100)
plt.savefig('accuracy.png')
plt.show()
plt.clf()
