import json
import matplotlib.pyplot as plt
import os

# All the data is organized in the following structure:
# Loss Type
# -> cifar10
# -> cifar100
# -> imagenette
# -> pretraining

# The pretraining folder contains the sampled hyperparams as well as loss (size N x 100) and rank (size N x 5 x 2) where rank is taken every 20 epochs and the first entry is entropic rank, the second is the robust rank metric

# The first three folders contain classification accuracies, each of the contains two subfolders with a linear projection head and a mlp projection head
# For each of the rank measurements (5 per run) a classification head was trained. Therefore each linear and mlp folder contains train and validation accuracies of size (N x 5).
# Losses should be (N x 5 x 20)

current_file_path = os.path.abspath(__file__)

vic1 = 'vicf1'
simclr1 = 'simclrf1'
dino1 = 'dinof1'

case = vic1

rank_path = case + '/pretraining/rank.json' 
cifar10acclin_path = case + '/cifar10/linear/val_accs.json'



# open the files
with open(rank_path, 'r') as file:
    ranks = json.load(file)

with open(cifar10acclin_path, 'r') as file:
    accs_cifar = json.load(file)



# this filters out the last entry of each (the one trained for 100 epochs)
rank = []
cifar = []


for i in range(len(ranks)):
    rank.append(ranks[i][-1][0])

for i in range(len(accs_cifar)):
    cifar.append(accs_cifar[i][-1])





plt.scatter(rank, cifar)
plt.xscale('log')  # Set the x-axis to be logarithmic

# Set labels and title
plt.xlabel('Rank (Logarithmic)')
plt.ylabel('Accuracy on Cifar10')
plt.title('Scatter Plot with Logarithmic X Axis')

# Show the plot
plt.show()