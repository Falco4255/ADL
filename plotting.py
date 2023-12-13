import json
import matplotlib.pyplot as plt
import os
import numpy as np

# All the data is organized in the following structure:
# Loss Type
# -> cifar10
# -> imagenette
# -> imagenette
# -> pretraining

# The pretraining folder contains the sampled hyperparams as well as loss (size N x 100) and rank (size N x 5 x 2) where rank is taken every 20 epochs and the first entry is entropic rank, the second is the robust rank metric

# The first three folders contain classification accuracies, each of the contains two subfolders with a linear projection head and a mlp projection head
# For each of the rank measurements (5 per run) a classification head was trained. Therefore each linear and mlp folder contains train and validation accuracies of size (N x 5).
# Losses should be (N x 6 x 20)






# open ranks
with open('simclrf1/pretraining/rank.json', 'r') as file:
    ranks_simclrf1 = np.array(json.load(file))

with open('simclrf2/pretraining/rank.json', 'r') as file:
    ranks_simclrf2 = np.array(json.load(file))

with open('vicf1/pretraining/rank.json', 'r') as file:
    ranks_vicf1 = np.array(json.load(file))

with open('vicf2/pretraining/rank.json', 'r') as file:
    ranks_vicf2 = np.array(json.load(file))

with open('dinof1/pretraining/rank.json', 'r') as file:
    ranks_dinof1 = np.array(json.load(file ))

with open('dinof2/pretraining/rank.json', 'r') as file:
    ranks_dinof2 = np.array(json.load(file))



# open linear cifar10 accs
with open('simclrf1/cifar10/linear/val_accs.json', 'r') as file:
    accs_lin_simclrf1_cifar10 = np.array(json.load(file))

with open('simclrf2/cifar10/linear/val_accs.json', 'r') as file:
    accs_lin_simclrf2_cifar10 = np.array(json.load(file))

with open('vicf1/cifar10/linear/val_accs.json', 'r') as file:
    accs_lin_vicf1_cifar10 = np.array(json.load(file))

with open('vicf2/cifar10/linear/val_accs.json', 'r') as file:
    accs_lin_vicf2_cifar10 = np.array(json.load(file))

with open('dinof1/cifar10/linear/val_accs.json', 'r') as file:
    accs_lin_dinof1_cifar10 = np.array(json.load(file))

with open('dinof2/cifar10/linear/val_accs.json', 'r') as file:
    accs_lin_dinof2_cifar10 = np.array(json.load(file))



# open mlp cifar10 accs
with open('simclrf1/cifar10/mlp/val_accs.json', 'r') as file:
    accs_mlp_simclrf1_cifar10 = np.array(json.load(file))

with open('simclrf2/cifar10/mlp/val_accs.json', 'r') as file:
    accs_mlp_simclrf2_cifar10 = np.array(json.load(file))

with open('vicf1/cifar10/mlp/val_accs.json', 'r') as file:
    accs_mlp_vicf1_cifar10 = np.array(json.load(file))

with open('vicf2/cifar10/mlp/val_accs.json', 'r') as file:
    accs_mlp_vicf2_cifar10 = np.array(json.load(file))

with open('dinof1/cifar10/mlp/val_accs.json', 'r') as file:
    accs_mlp_dinof1_cifar10 = np.array(json.load(file))

with open('dinof2/cifar10/mlp/val_accs.json', 'r') as file:
    accs_mlp_dinof2_cifar10 = np.array(json.load(file))



# open linear cifar100 accs
with open('simclrf1/cifar100/linear/val_accs.json', 'r') as file:
    accs_lin_simclrf1_cifar100 = np.array(json.load(file))

with open('simclrf2/cifar100/linear/val_accs.json', 'r') as file:
    accs_lin_simclrf2_cifar100 = np.array(json.load(file))

with open('vicf1/cifar100/linear/val_accs.json', 'r') as file:
    accs_lin_vicf1_cifar100 = np.array(json.load(file))

with open('vicf2/cifar100/linear/val_accs.json', 'r') as file:
    accs_lin_vicf2_cifar100 = np.array(json.load(file))

with open('dinof1/cifar100/linear/val_accs.json', 'r') as file:
    accs_lin_dinof1_cifar100 = np.array(json.load(file))

with open('dinof2/cifar100/linear/val_accs.json', 'r') as file:
    accs_lin_dinof2_cifar100 = np.array(json.load(file))



# open mlp cifar100 accs
with open('simclrf1/cifar100/mlp/val_accs.json', 'r') as file:
    accs_mlp_simclrf1_cifar100 = np.array(json.load(file))

with open('simclrf2/cifar100/mlp/val_accs.json', 'r') as file:
    accs_mlp_simclrf2_cifar100 = np.array(json.load(file))

with open('vicf1/cifar100/mlp/val_accs.json', 'r') as file:
    accs_mlp_vicf1_cifar100 = np.array(json.load(file))

with open('vicf2/cifar100/mlp/val_accs.json', 'r') as file:
    accs_mlp_vicf2_cifar100 = np.array(json.load(file))

with open('dinof1/cifar100/mlp/val_accs.json', 'r') as file:
    accs_mlp_dinof1_cifar100 = np.array(json.load(file))

with open('dinof2/cifar100/mlp/val_accs.json', 'r') as file:
    accs_mlp_dinof2_cifar100 = np.array(json.load(file))


# open linear imagenette accs
with open('simclrf1/imagenette/linear/val_accs.json', 'r') as file:
    accs_lin_simclrf1_imagenette = np.array(json.load(file))

with open('simclrf2/imagenette/linear/val_accs.json', 'r') as file:
    accs_lin_simclrf2_imagenette = np.array(json.load(file))

with open('vicf1/imagenette/linear/val_accs.json', 'r') as file:
    accs_lin_vicf1_imagenette = np.array(json.load(file))

with open('vicf2/imagenette/linear/val_accs.json', 'r') as file:
    accs_lin_vicf2_imagenette = np.array(json.load(file))

with open('dinof1/imagenette/linear/val_accs.json', 'r') as file:
    accs_lin_dinof1_imagenette = np.array(json.load(file))

with open('dinof2/imagenette/linear/val_accs.json', 'r') as file:
    accs_lin_dinof2_imagenette = np.array(json.load(file))



# open mlp imagenette accs
with open('simclrf1/imagenette/mlp/val_accs.json', 'r') as file:
    accs_mlp_simclrf1_imagenette = np.array(json.load(file))

with open('simclrf2/imagenette/mlp/val_accs.json', 'r') as file:
    accs_mlp_simclrf2_imagenette = np.array(json.load(file))

with open('vicf1/imagenette/mlp/val_accs.json', 'r') as file:
    accs_mlp_vicf1_imagenette = np.array(json.load(file))

with open('vicf2/imagenette/mlp/val_accs.json', 'r') as file:
    accs_mlp_vicf2_imagenette = np.array(json.load(file))

with open('dinof1/imagenette/mlp/val_accs.json', 'r') as file:
    accs_mlp_dinof1_imagenette = np.array(json.load(file))

with open('dinof2/imagenette/mlp/val_accs.json', 'r') as file:
    accs_mlp_dinof2_imagenette = np.array(json.load(file))


with open('simclrf1/pretraining/loss.json', 'r') as file:
    loss_mean_simclrf1 = np.mean(np.array(json.load(file)),axis=0)

with open('simclrf2/pretraining/loss.json', 'r') as file:
    loss_mean_simclrf2 = np.mean(np.array(json.load(file)),axis=0)

with open('vicf1/pretraining/loss.json', 'r') as file:
    loss_mean_vicf1 = np.mean(np.array(json.load(file)),axis=0)

with open('vicf2/pretraining/loss.json', 'r') as file:
    loss_mean_vicf2 = np.mean(np.array(json.load(file)),axis=0)

with open('dinof1/pretraining/loss.json', 'r') as file:
    loss_mean_dinof1 = np.mean(np.array(json.load(file)),axis=0)

with open('dinof2/pretraining/loss.json', 'r') as file:
    loss_mean_dinof2 = np.mean(np.array(json.load(file)),axis=0)



# Plot of all normal models on cifar10, next to cifar100 next to imagenette


def first_plot():
        # Creating three subplots
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))  # 1 row, 3 columns

        # Imagenette
        axs[0].scatter(ranks_simclrf1[:,-1,0], accs_lin_simclrf1_imagenette[:,-1],label='SimCLR Linear', marker='o',color = 'r')
        axs[0].scatter(ranks_simclrf1[:,-1,0], accs_mlp_simclrf1_imagenette[:,-1],label='SimCLR MLP', marker='^',color = 'salmon')
        axs[0].scatter(ranks_vicf1[:,-1,0], accs_lin_vicf1_imagenette[:,-1],label='VICReg Linear', marker='o',color = 'b')
        axs[0].scatter(ranks_vicf1[:,-1,0], accs_mlp_vicf1_imagenette[:,-1],label='VICReg MLP', marker='^',color = 'cyan')
        axs[0].scatter(ranks_dinof1[:,-1,0], accs_lin_dinof1_imagenette[:,-1],label='DINO Linear', marker='o',color = 'g')
        axs[0].scatter(ranks_dinof1[:,-1,0], accs_mlp_dinof1_imagenette[:,-1],label='DINO MLP', marker='^',color = 'palegreen')
        axs[0].set_title('IMAGENETTE',fontsize=16)
        axs[0].set_xlabel('Entropic Rank',fontsize=14)
        axs[0].set_ylabel('Validation Accuracy',fontsize=14)

        axs[1].scatter(ranks_simclrf1[:,-1,0], accs_lin_simclrf1_cifar10[:,-1],label='SimCLR Linear', marker='o',color = 'r')
        axs[1].scatter(ranks_simclrf1[:,-1,0], accs_mlp_simclrf1_cifar10[:,-1],label='SimCLR MLP', marker='^',color = 'salmon')
        axs[1].scatter(ranks_vicf1[:,-1,0], accs_lin_vicf1_cifar10[:,-1],label='VICReg Linear', marker='o',color = 'b')
        axs[1].scatter(ranks_vicf1[:,-1,0], accs_mlp_vicf1_cifar10[:,-1],label='VICReg MLP', marker='^',color = 'cyan')
        axs[1].scatter(ranks_dinof1[:,-1,0], accs_lin_dinof1_cifar10[:,-1],label='DINO Linear', marker='o',color = 'g')
        axs[1].scatter(ranks_dinof1[:,-1,0], accs_mlp_dinof1_cifar10[:,-1],label='DINO MLP', marker='^',color = 'palegreen')
        axs[1].set_title('CIFAR10',fontsize=16)
        axs[1].set_xlabel('Entropic Rank',fontsize=14)


        axs[2].scatter(ranks_simclrf1[:,-1,0], accs_lin_simclrf1_cifar100[:,-1],label='SimCLR Linear', marker='o',color = 'r')
        axs[2].scatter(ranks_simclrf1[:,-1,0], accs_mlp_simclrf1_cifar100[:,-1],label='SimCLR MLP', marker='^',color = 'salmon')
        axs[2].scatter(ranks_vicf1[:,-1,0], accs_lin_vicf1_cifar100[:,-1],label='VICReg Linear', marker='o',color = 'b')
        axs[2].scatter(ranks_vicf1[:,-1,0], accs_mlp_vicf1_cifar100[:,-1],label='VICReg MLP', marker='^',color = 'cyan')
        axs[2].scatter(ranks_dinof1[:,-1,0], accs_lin_dinof1_cifar100[:,-1],label='DINO Linear', marker='o',color = 'g')
        axs[2].scatter(ranks_dinof1[:,-1,0], accs_mlp_dinof1_cifar100[:,-1],label='DINO MLP', marker='^',color = 'palegreen')
        axs[2].set_title('CIFAR100',fontsize=16)
        axs[2].set_xlabel('Entropic Rank',fontsize=14)


        for ax in axs:
            ax.set_xscale('log')

        handles, labels = axs[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='center left', bbox_to_anchor=(0.01, 0.5), title='Model Type',fontsize=14, title_fontsize=14)


        plt.tight_layout()  # Adjusts subplot parameters to give specified 
        plt.subplots_adjust(left=0.21,wspace=0.2)
        plt.show()

def second_plot():
        #plot of imagenette models over time, simclr, next to vic, next to dino
        # Creating three subplots
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))  # 1 row, 3 columns

        axs[0].scatter(ranks_simclrf1[:,0,0], accs_mlp_simclrf1_imagenette[:,0],label='20 Epochs', marker='o',color = 'salmon')
        axs[0].scatter(ranks_simclrf1[:,1,0], accs_mlp_simclrf1_imagenette[:,1],label='40 Epochs', marker='o',color = 'g')
        axs[0].scatter(ranks_simclrf1[:,2,0], accs_mlp_simclrf1_imagenette[:,2],label='60 Epochs', marker='o',color = 'b')
        axs[0].scatter(ranks_simclrf1[:,3,0], accs_mlp_simclrf1_imagenette[:,3],label='80 Epochs', marker='o',color = 'cyan')
        axs[0].scatter(ranks_simclrf1[:,4,0], accs_mlp_simclrf1_imagenette[:,4],label='100 Epochs', marker='o',color = 'r')
        axs[0].set_title('SimCLR',fontsize=16)
        axs[0].set_xlabel('Entropic Rank',fontsize=14)
        axs[0].set_ylabel('Validation Accuracy',fontsize=14)


        axs[1].scatter(ranks_vicf1[:,0,0], accs_mlp_vicf1_imagenette[:,0],label='20 Epochs', marker='o',color = 'salmon')
        axs[1].scatter(ranks_vicf1[:,1,0], accs_mlp_vicf1_imagenette[:,1],label='40 Epochs', marker='o',color = 'g')
        axs[1].scatter(ranks_vicf1[:,2,0], accs_mlp_vicf1_imagenette[:,2],label='60 Epochs', marker='o',color = 'b')
        axs[1].scatter(ranks_vicf1[:,3,0], accs_mlp_vicf1_imagenette[:,3],label='80 Epochs', marker='o',color = 'cyan')
        axs[1].scatter(ranks_vicf1[:,4,0], accs_mlp_vicf1_imagenette[:,4],label='100 Epochs', marker='o',color = 'r')
        axs[1].set_title('VICReg',fontsize=16)
        axs[1].set_xlabel('Entropic Rank',fontsize=14)

        axs[2].scatter(ranks_dinof1[:,0,0], accs_mlp_dinof1_imagenette[:,0],label='20 Epochs', marker='o',color = 'salmon')
        axs[2].scatter(ranks_dinof1[:,1,0], accs_mlp_dinof1_imagenette[:,1],label='40 Epochs', marker='o',color = 'g')
        axs[2].scatter(ranks_dinof1[:,2,0], accs_mlp_dinof1_imagenette[:,2],label='60 Epochs', marker='o',color = 'b')
        axs[2].scatter(ranks_dinof1[:,3,0], accs_mlp_dinof1_imagenette[:,3],label='80 Epochs', marker='o',color = 'cyan')
        axs[2].scatter(ranks_dinof1[:,4,0], accs_mlp_dinof1_imagenette[:,4],label='100 Epochs', marker='o',color = 'r')
        axs[2].set_title('DINO',fontsize=16)
        axs[2].set_xlabel('Entropic Rank',fontsize=14)

        

        for ax in axs:
            ax.set_xscale('log')

        handles, labels = axs[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='center left', bbox_to_anchor=(0.01, 0.5), title='Pretraining Epochs',fontsize=14, title_fontsize=14)


        plt.tight_layout()  # Adjusts subplot parameters to give specified 
        plt.subplots_adjust(left=0.2,wspace=0.2)
        plt.show()

def third_plot():
        # Creating three subplots
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))  # 1 row, 3 columns

        # Imagenette
        axs[0].scatter(ranks_simclrf1[:,-1,0], accs_mlp_simclrf1_imagenette[:,-1],label='SimCLR Base', marker='o',color = 'r')
        axs[0].scatter(ranks_simclrf2[:,-1,0], accs_mlp_simclrf2_imagenette[:,-1],label='SimCLR Extended', marker='^',color = 'salmon')
        axs[0].scatter(ranks_vicf1[:,-1,0], accs_mlp_vicf1_imagenette[:,-1],label='VICReg Base', marker='o',color = 'b')
        axs[0].scatter(ranks_vicf2[:,-1,0], accs_mlp_vicf2_imagenette[:,-1],label='VICReg Extended', marker='^',color = 'cyan')
        axs[0].scatter(ranks_dinof1[:,-1,0], accs_mlp_dinof1_imagenette[:,-1],label='DINO Base', marker='o',color = 'g')
        axs[0].scatter(ranks_dinof2[:,-1,0], accs_mlp_dinof2_imagenette[:,-1],label='DINO Extended', marker='^',color = 'palegreen')
        axs[0].set_title('IMAGENETTE',fontsize=16)
        axs[0].set_xlabel('Entropic Rank',fontsize=14)
        axs[0].set_ylabel('Validation Accuracy',fontsize=14)

        axs[1].scatter(ranks_simclrf1[:,-1,0], accs_mlp_simclrf1_cifar10[:,-1],label='SimCLR Base', marker='o',color = 'r')
        axs[1].scatter(ranks_simclrf2[:,-1,0], accs_mlp_simclrf2_cifar10[:,-1],label='SimCLR Extended', marker='^',color = 'salmon')
        axs[1].scatter(ranks_vicf1[:,-1,0], accs_mlp_vicf1_cifar10[:,-1],label='VICReg Base', marker='o',color = 'b')
        axs[1].scatter(ranks_vicf2[:,-1,0], accs_mlp_vicf2_cifar10[:,-1],label='VICReg Extended', marker='^',color = 'cyan')
        axs[1].scatter(ranks_dinof1[:,-1,0], accs_mlp_dinof1_cifar10[:,-1],label='DINO Base', marker='o',color = 'g')
        axs[1].scatter(ranks_dinof2[:,-1,0], accs_mlp_dinof2_cifar10[:,-1],label='DINO Extended', marker='^',color = 'palegreen')
        axs[1].set_title('CIFAR10',fontsize=16)
        axs[1].set_xlabel('Entropic Rank',fontsize=14)

        axs[2].scatter(ranks_simclrf1[:,-1,0], accs_mlp_simclrf1_cifar100[:,-1],label='SimCLR Base', marker='o',color = 'r')
        axs[2].scatter(ranks_simclrf2[:,-1,0], accs_mlp_simclrf2_cifar100[:,-1],label='SimCLR Extended', marker='^',color = 'salmon')
        axs[2].scatter(ranks_vicf1[:,-1,0], accs_mlp_vicf1_cifar100[:,-1],label='VICReg Base', marker='o',color = 'b')
        axs[2].scatter(ranks_vicf2[:,-1,0], accs_mlp_vicf2_cifar100[:,-1],label='VICReg Extended', marker='^',color = 'cyan')
        axs[2].scatter(ranks_dinof1[:,-1,0], accs_mlp_dinof1_cifar100[:,-1],label='DINO Base', marker='o',color = 'g')
        axs[2].scatter(ranks_dinof2[:,-1,0], accs_mlp_dinof2_cifar100[:,-1],label='DINO Extended', marker='^',color = 'palegreen')
        axs[2].set_title('CIFAR100',fontsize=16)
        axs[2].set_xlabel('Entropic Rank',fontsize=14)


        for ax in axs:
            ax.set_xscale('log')

        handles, labels = axs[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='center left', bbox_to_anchor=(0.01, 0.5), title='Model Type',fontsize=14, title_fontsize=14)


        plt.tight_layout()  # Adjusts subplot parameters to give specified 
        plt.subplots_adjust(left=0.23,wspace=0.2)
        plt.show()

#rank comparison
def fourth_plot():
        # Creating three subplots
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))  # 1 row, 3 columns

        # Imagenette
        axs[0].scatter(ranks_simclrf1[:,0,0], ranks_simclrf1[:,0,1],label='20 Epochs', marker='o',color = 'salmon')
        axs[0].scatter(ranks_simclrf1[:,1,0], ranks_simclrf1[:,1,1],label='40 Epochs', marker='o',color = 'g')
        axs[0].scatter(ranks_simclrf1[:,2,0], ranks_simclrf1[:,2,1],label='60 Epochs', marker='o',color = 'b')
        axs[0].scatter(ranks_simclrf1[:,3,0], ranks_simclrf1[:,3,1],label='80 Epochs', marker='o',color = 'cyan')
        axs[0].scatter(ranks_simclrf1[:,4,0], ranks_simclrf1[:,4,1],label='100 Epochs', marker='o',color = 'r')
        axs[0].set_title('SimCLR Rank Comparison',fontsize=16)
        axs[0].set_xlabel('Entropic Rank',fontsize=14)
        axs[0].set_ylabel('Robust Rank',fontsize=14)

        axs[1].scatter(ranks_vicf1[:,0,0], ranks_vicf1[:,0,1],label='20 Epochs', marker='o',color = 'salmon')
        axs[1].scatter(ranks_vicf1[:,1,0], ranks_vicf1[:,1,1],label='40 Epochs', marker='o',color = 'g')
        axs[1].scatter(ranks_vicf1[:,2,0], ranks_vicf1[:,2,1],label='60 Epochs', marker='o',color = 'b')
        axs[1].scatter(ranks_vicf1[:,3,0], ranks_vicf1[:,3,1],label='80 Epochs', marker='o',color = 'cyan')
        axs[1].scatter(ranks_vicf1[:,4,0], ranks_vicf1[:,4,1],label='100 Epochs', marker='o',color = 'r')
        axs[1].set_title('VICReg Rank Comparison',fontsize=16)
        axs[1].set_xlabel('Entropic Rank',fontsize=14)


        axs[2].scatter(ranks_dinof1[:,0,0], ranks_dinof1[:,0,1],label='20 Epochs', marker='o',color = 'salmon')
        axs[2].scatter(ranks_dinof1[:,1,0], ranks_dinof1[:,1,1],label='40 Epochs', marker='o',color = 'g')
        axs[2].scatter(ranks_dinof1[:,2,0], ranks_dinof1[:,2,1],label='60 Epochs', marker='o',color = 'b')
        axs[2].scatter(ranks_dinof1[:,3,0], ranks_dinof1[:,3,1],label='80 Epochs', marker='o',color = 'cyan')
        axs[2].scatter(ranks_dinof1[:,4,0], ranks_dinof1[:,4,1],label='100 Epochs', marker='o',color = 'r')
        axs[2].set_title('DINO Rank Comparison',fontsize=16)
        axs[2].set_xlabel('Entropic Rank',fontsize=14)

        # for ax in axs:
        #     ax.set_xscale('log')
        #     ax.set_yscale('log')

        handles, labels = axs[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='center left', bbox_to_anchor=(0.01, 0.5), title='Pretraining Epochs',fontsize=14, title_fontsize=14)


        plt.tight_layout()  # Adjusts subplot parameters to give specified 
        plt.subplots_adjust(left=0.2,wspace=0.2)
        plt.show()

def main():
    first_plot()
    second_plot()
    third_plot()
    fourth_plot()

if __name__ == '__main__':
    main()
    
