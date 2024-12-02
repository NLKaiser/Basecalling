import matplotlib.pyplot as plt
import numpy as np
import ast

epoch = []
train_loss = []
val_loss = []
val_mean_accuracy = []
val_median_accuracy = []
lru_values = []

with open("training.csv") as f:
    next(f)
    for line in f:
        l = line.split(";")
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
plt.axhline(y=200, color='green', linestyle='-', linewidth=1, label='y = 200')
plt.axhline(y=100, color='greenyellow', linestyle='-', linewidth=1, label='y = 100')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.xlim(0, len(epoch))
plt.ylim(0, max(max(train_loss), max(val_loss)))
plt.savefig('loss.png')
plt.show()
plt.clf()
plt.close()

plt.plot(epoch, val_mean_accuracy, label='Mean accuracy', color='orange', marker='o', markersize=6, linestyle='None')
plt.plot(epoch, val_median_accuracy, label='Median accuracy', color='grey', marker='+', markersize=4, linestyle='None')
plt.axhline(y=85, color='green', linestyle='-', linewidth=1, label='y = 85')
plt.axhline(y=95, color='greenyellow', linestyle='-', linewidth=1, label='y = 95')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.xlim(0, len(epoch))
plt.ylim(0, 100)
plt.savefig('accuracy.png')
plt.show()
plt.clf()
plt.close()


def plot_lru_unit_circle(title, nu_log, theta_log):
    fig, ax = plt.subplots(figsize=(6, 6))
    # Plot the unit circle
    circle = plt.Circle((0, 0), 1, color='black', fill=False, linestyle='-')
    ax.add_artist(circle)

    # Iterate over points and plot
    for nu, theta in zip(nu_log, theta_log):
        L = np.exp(-np.exp(nu) + 1j * np.exp(theta))
        L_real = np.real(L)
        L_imag = np.imag(L)
        
        # Plot the points as red dots
        ax.plot(L_real, L_imag, color='#ff00d0', marker='o', markersize=4)

    # Set aspect ratio and limits
    ax.set_aspect('equal', 'box')
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)

    # Add grid and labels
    ax.grid(True)
    ax.set_xlabel('Real')
    ax.set_ylabel('Imaginary')
    ax.set_title(title)
    
    title = title.replace(" ", "_").replace(";", "")
    plt.savefig(title + '.png')
    plt.show()
    plt.clf()
    plt.close()

def plot_lru_epoch(epoch):
    with open("training.csv") as f:
        next(f)
        for i, line in enumerate(f):
            if i == epoch:
                l = line.split(";")
                lru_values = ast.literal_eval(l[6])
    for layer in lru_values.keys():
        title = f"Epoch {epoch+1}; Layer {layer+1};"
        plot_lru_unit_circle(title + " forward", lru_values[layer]["nu_fw"], lru_values[layer]["theta_fw"])
        plot_lru_unit_circle(title + " reverse", lru_values[layer]["nu_rv"], lru_values[layer]["theta_rv"])

plot_lru_epoch(0)
plot_lru_epoch(epoch[-1])

def plot_alignments(query1, ref1, query2, ref2):
    def get_match_array(query, ref):
        return np.array([
            1 if b1 == b2 and b1 != '-' else 
            2 if b1 != b2 and b1 != '-' and b2 != '-' else 
            0
            for b1, b2 in zip(query, ref)
        ])
    
    # Generate match arrays for both alignments
    match_array1 = get_match_array(query1, ref1)
    match_array2 = get_match_array(query2, ref2)
    
    # Create the plot with two subplots
    fig, axs = plt.subplots(2, 1, figsize=(15, 5), sharex=False)  # Two subplots, shared x-axis
    
    # Colors for match (green), substitution (orange), and gap (red)
    colors = ['red', 'green', 'orange']
    
    def subplot(match_array, i, title):
        for value, color in enumerate(colors):
            indices = np.where(match_array == value)[0]
            axs[i].bar(indices, [1] * len(indices), color=color, width=1, label=f"{['Gap', 'Match', 'Substitution'][value]}")
        axs[i].set_title(title)
        axs[i].set_ylim(0, 1.5)
        axs[i].set_yticks([])
        axs[i].legend(loc="upper right", fontsize=8)
    
    subplot(match_array1, 0, "Local")
    subplot(match_array2, 1, "Global")
    
    # Add shared x-axis label
    axs[1].set_xlabel("Alignment Position")
    
    plt.tight_layout()
    plt.savefig("alignments.png")
    plt.show()
    plt.clf()
    plt.close()

def plot_alignments_epoch(epoch):
    with open("training.csv") as f:
        next(f)
        for i, line in enumerate(f):
            if i == epoch:
                l = line.split(";")
                alignments = ast.literal_eval(l[7])
    plot_alignments(alignments["pred_original"], alignments["ref_original"],
        alignments["pred_global"], alignments["ref_global"])

plot_alignments_epoch(epoch[-1])
