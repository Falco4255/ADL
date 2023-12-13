import sys
from PIL import Image, ImageEnhance
import random
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.transforms.functional as Fvis
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
import torchvision
from torchvision.datasets import CIFAR10,  ImageFolder
from torchvision.models import resnet18
from torchvision.models.resnet import BasicBlock
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from tqdm import tqdm
from pathlib import Path
from lightly import loss as lightly_loss
import torchlars
from fastai.vision.all import URLs, untar_data
import json
from torchvision.datasets import CIFAR10, CIFAR100


from custom_transforms import CustomTransforms
from custom_models import *
from utils import *




# Linear warm-up scheduler
def linear_warmup(current_epoch,warmup_epochs):
    return current_epoch / warmup_epochs


def download_imagenette(path = 'datasets/imagenette'):
        
        # Create folder 'datasets'
        Path('datasets').mkdir(parents=True, exist_ok=True)

        # Specify the folder where you want to download Imagenette
        data_dir = path

        # Download and untar the Imagenette dataset
        path = untar_data(URLs.IMAGENETTE_160)

        # Move the contents of the downloaded directory to your desired location
        path.rename(data_dir)




def get_pretrain_loaders(path,batch_size):
     pretrain_dataset = ImageFolder(root=path + '/train', transform=CustomTransforms(is_pretrain=True, is_val=False))
     val_dataset = ImageFolder(root=path + '/val', transform=CustomTransforms(is_pretrain=False, is_val=True))
     rank_dataset = ImageFolder(root=path + '/train', transform=CustomTransforms(is_pretrain=False, is_val=True))
     dino_dataset = ImageFolder(root=path + '/train', transform=CustomTransforms(is_pretrain=False, is_val=False,is_classification=False,is_dino=True))

     pretrain_loader = DataLoader(dataset=pretrain_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
     val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
     rank_loader = DataLoader(dataset=rank_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
     dino_loader = DataLoader(dataset=dino_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
     return [pretrain_loader, val_loader, rank_loader,dino_loader]


def get_classification_loadersCIFAR10(data_path,batch_size):
    train_dataset = CIFAR10(root=data_path, train=True, download=True, transform=CustomTransforms(is_pretrain=False, is_val=False,is_classification=True,is_dino=False))
    eval_dataset = CIFAR10(root=data_path, train=True, download=True, transform=CustomTransforms(is_pretrain=False, is_val=True,is_classification=False,is_dino=False))
    val_dataset = CIFAR10(root=data_path, train=False, download=True, transform=CustomTransforms(is_pretrain=False, is_val=True,is_classification=False,is_dino=False))

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    eval_loader = DataLoader(dataset=eval_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    return [train_loader, val_loader, eval_loader]


def get_classification_loadersCIFAR100(data_path,batch_size):
    train_dataset = CIFAR100(root=data_path, train=True, download=True, transform=CustomTransforms(is_pretrain=False, is_val=False,is_classification=True,is_dino=False))
    eval_dataset = CIFAR100(root=data_path, train=True, download=True, transform=CustomTransforms(is_pretrain=False, is_val=True,is_classification=False,is_dino=False))
    val_dataset = CIFAR100(root=data_path, train=False, download=True, transform=CustomTransforms(is_pretrain=False, is_val=True,is_classification=False,is_dino=False))

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    eval_loader = DataLoader(dataset=eval_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    return [train_loader, val_loader, eval_loader]

def get_classification_loaders(path,batch_size):
    train_dataset = ImageFolder(root=path + '/train', transform=CustomTransforms(is_pretrain=False, is_val=False,is_classification=True,is_dino=False))
    eval_dataset = ImageFolder(root=path + '/train', transform=CustomTransforms(is_pretrain=False, is_val=True,is_classification=False,is_dino=False))
    val_dataset = ImageFolder(root=path + '/val', transform=CustomTransforms(is_pretrain=False, is_val=True,is_classification=False,is_dino=False))

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    eval_loader = DataLoader(dataset=eval_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    return [train_loader, val_loader, eval_loader]

def dino_pretraining_run(lr,wd,teacher_temp,student_temp,batch_size,epochs,data_loaders,run,num_runs,extended_head = False):
     pretrain_loader, val_loader, rank_loader, dino_loader = data_loaders


    #Architecture information taken from Emerging Properties in Self-Supervised Vision Transformers https://arxiv.org/abs/2104.14294
     resnet_s = torchvision.models.resnet18()
     backbone_s = nn.Sequential(*list(resnet_s.children())[:-1])
     if extended_head:
        proj_head_s = ProjectionHead(512,[2048,2048,2048],512)
     else:
        proj_head_s = ProjectionHead(512,[2048,2048],512)
     dino_head_s = DINOHead(512,8192)
     pretrain_model_s = PretrainModel(backbone_s,proj_head_s)
     dino_model_s = DINOModel(backbone_s,proj_head_s,dino_head_s)
     dino_model_s.to(device)

     resnet_t = torchvision.models.resnet18()
     backbone_t = nn.Sequential(*list(resnet_t.children())[:-1])
     if extended_head:
        proj_head_t = ProjectionHead(512,[2048,2048,2048],512)
     else:
        proj_head_t = ProjectionHead(512,[2048,2048],512)
     dino_head_t = DINOHead(512,8192)
     pretrain_model_t = PretrainModel(backbone_t,proj_head_t)
     dino_model_t = DINOModel(backbone_t,proj_head_t,dino_head_t)
     dino_model_t.to(device)

     for param in dino_model_t.parameters():
         param.requires_grad = False

     m = 0.9 #center momentum
     # AdamW imported from pytorch
     optimizer = torch.optim.AdamW(dino_model_s.parameters(), lr=lr, weight_decay=wd)
     criterion = DINOLoss(teacher_temp,student_temp,m,8192)
     criterion.to(device)

     avg_losses = []
     rankme = []
     models_t = []
     backbones_t = []
     models_s = []
     backbones_s = []



     lmin = torch.tensor(0.996, dtype=torch.float)
     lmax = torch.tensor(1, dtype=torch.float)
     #custom Cosine Schedule
     l_schedule = CosineSchedule(lmin,lmax,torch.tensor(epochs,dtype=float))
     l_schedule.to(device)

     pretrain_model_t.eval()
     rankme.append(RankME(pretrain_model_t,rank_loader))
     pretrain_model_t.train()

     print("Starting Training")
     for epoch in range(epochs):
        total_loss = 0
        l = l_schedule.step()
        for batch in tqdm(dino_loader, desc=f'Training run {run + 1}/{num_runs}', leave=False):
            views = batch[0]
            glob_views = [views[0].to(device),views[1].to(device)]
            views = [views[i].to(device) for i in range(len(views))]
            z_t = [dino_model_t(glob_views[i])for i in range(len(glob_views))]
            z_s = [dino_model_s(views[i]) for i in range(len(views))]
            loss = criterion(z_t,z_s)
            total_loss += loss.item()
            optimizer.zero_grad()

           
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                for params, paramt in zip(dino_model_s.parameters(), dino_model_t.parameters()):
                    paramt.data = (1.0 - l) * params.data + l * paramt.data

            

        avg_loss = total_loss / len(pretrain_loader)
        avg_losses.append(avg_loss)


        if (epoch + 1)%20 == 0:
            pretrain_model_t.eval()
            rankme.append(RankME(pretrain_model_t,rank_loader))
            pretrain_model_t.train()
            models_t.append(dino_model_t.state_dict())
            models_s.append(dino_model_s.state_dict())
            backbones_t.append(backbone_t.state_dict())
            backbones_s.append(backbone_s.state_dict())

     return rankme, avg_losses, models_t,models_s,backbones_t,backbones_s


def vicreg_pretraining_run(lr,weight_decay,eta,lamb,mu,batch_size,epochs,data_loaders,run,num_runs, extended_head = False):
     pretrain_loader, val_loader, rank_loader,_ = data_loaders

     resnet = torchvision.models.resnet18()
     backbone = nn.Sequential(*list(resnet.children())[:-1])
     if extended_head:
         proj_head = ProjectionHead(512,[2048,2048,2048],512)
     else:
        proj_head = ProjectionHead(512,[2048,2048],512)
     pretrain_model = PretrainModel(backbone,proj_head)
     pretrain_model.to(device)

     #vic loss taken from the lightly library
     criterion = lightly_loss.vicreg_loss.VICRegLoss(lambda_param = lamb, mu_param = mu, nu_param = eta) #from lightly
     base_optimizer = torch.optim.SGD(pretrain_model.parameters(), lr=lr, weight_decay=weight_decay)

     #LARS optimizer taken from the torchlars library
     optimizer = torchlars.LARS(optimizer=base_optimizer, trust_coef=0.001)

     avg_losses = []
     rankme = []
     models = []
     backbones = []

     pretrain_model.eval()
     rankme.append(RankME(pretrain_model,rank_loader))
     pretrain_model.train()

     for epoch in range(epochs):
        total_loss = 0
        for batch in tqdm(pretrain_loader, desc=f'VICREG run {run + 1}/{num_runs}', leave=False):
            x0, x1 = batch[0]
            x0 = x0.to(device)
            x1 = x1.to(device)
            z0 = pretrain_model(x0)
            z1 = pretrain_model(x1)
            loss = criterion(z0, z1)
            total_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        avg_loss = total_loss / len(pretrain_loader)
        avg_losses.append(avg_loss)

        if (epoch + 1)%20 == 0:
            pretrain_model.eval()
            rankme.append(RankME(pretrain_model,rank_loader))
            pretrain_model.train()
            models.append(pretrain_model.state_dict())
            backbones.append(backbone.state_dict())

     return rankme, avg_losses, models, backbones



def simclr_pretraining_run(lr,wd,temp,batch_size,epochs,data_loaders,run,num_runs, extended_head = False):
     pretrain_loader, val_loader, rank_loader,_ = data_loaders

     resnet = torchvision.models.resnet18()
     backbone = nn.Sequential(*list(resnet.children())[:-1])
     if extended_head:
         proj_head = ProjectionHead(512,[2048,2048,2048],512)
     else:
        proj_head = ProjectionHead(512,[2048,2048],512)
     pretrain_model = PretrainModel(backbone,proj_head)
     pretrain_model.to(device)

     #NTXentLoss taken from the lightly library
     criterion = lightly_loss.NTXentLoss(temperature=temp)
     base_optimizer = torch.optim.SGD(pretrain_model.parameters(), lr=lr, weight_decay=wd)

     #LARS optimizer taken from the torchlars library
     optimizer = torchlars.LARS(optimizer=base_optimizer, trust_coef=0.001)

     avg_losses = []
     rankme = []
     models = []
     backbones = []

     pretrain_model.eval()
     rankme.append(RankME(pretrain_model,rank_loader))
     pretrain_model.train()

     for epoch in range(epochs):
        total_loss = 0
        for batch in tqdm(pretrain_loader, desc=f'SIMCLR run {run + 1}/{num_runs}', leave=False):
            x0, x1 = batch[0]
            x0 = x0.to(device)
            x1 = x1.to(device)
            z0 = pretrain_model(x0)
            z1 = pretrain_model(x1)
            loss = criterion(z0, z1)
            total_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        avg_loss = total_loss / len(pretrain_loader)
        avg_losses.append(avg_loss)

        if (epoch + 1)%20 == 0:
            pretrain_model.eval()
            rankme.append(RankME(pretrain_model,rank_loader))
            pretrain_model.train()
            models.append(pretrain_model.state_dict())
            backbones.append(backbone.state_dict())

     return rankme, avg_losses, models, backbones


def dino_pretraining_routine(num_runs = 1,identifier='dino', extended_head = False):
     path = 'pretraining/'+identifier
     model_path = path +'/models'
     Path(model_path).mkdir(parents=True, exist_ok=True)
     
     wd = 1e-6
     tpt_min = 0.01
     tpt_max = 0.07
     tps_min = 0.1
     tps_max = 0.5
     batch_size = 64 # vram at limit
     lr = 1e-3

     epochs = 100
     

     data_path = 'datasets/imagenette'
     data_loaders = get_pretrain_loaders(data_path,batch_size)

     rank_array = []
     loss_array = []
     tpts = []
     tpss = []

     for run in range(num_runs):
          run_path = model_path+'/run'+str(run)
          Path(run_path).mkdir(parents=True, exist_ok=True)
          tpt = sample_linear(tpt_min,tpt_max)
          tps = sample_linear(tps_min,tps_max)
          tpts.append(tpt)
          tpss.append(tps)
          new_rank_array, new_loss_array, models_t,models_s,backbones_t,backbones_s = dino_pretraining_run(lr,wd,tpt,tps,batch_size,epochs,data_loaders,run,num_runs,extended_head)
          
          epochstring = ['20','40','60','80','100']
          for i in range(len(models_t)):
            torch.save(models_t[i], run_path+'/'+epochstring[i]+'model_t.pth')
            torch.save(models_s[i], run_path+'/'+epochstring[i]+'model_s.pth')
            torch.save(backbones_t[i],run_path+'/'+epochstring[i]+'backbone_t.pth')
            torch.save(backbones_s[i],run_path+'/'+epochstring[i]+'backbone_s.pth')

          rank_array.append(new_rank_array)
          loss_array.append(new_loss_array)
     with open(path+'/tpt.json', 'w') as file:
        json.dump(tpts, file)
     with open(path+'/tps.json', 'w') as file:
        json.dump(tpss, file)
     with open(path+'/rank.json', 'w') as file:
        json.dump(rank_array, file)
     with open(path+'/loss.json', 'w') as file:
        json.dump(loss_array, file)

def vicreg_pretraining_routine(num_runs = 1,identifier='vicreg', extended_head = False):
     path = 'pretraining/'+identifier
     model_path = path +'/models'
     Path(model_path).mkdir(parents=True, exist_ok=True)
     
     lr_min = 0.1
     lr_max = 0.5
     weight_decay = 1e-6
     lambda_min = 5
     lambda_max = 50
     mu = 25
     eta_min = 0.25
     eta_max = 16
     batch_size = 256 #significantly smaller dataset
     epochs = 100

     data_path = 'datasets/imagenette'
     data_loaders = get_pretrain_loaders(data_path,batch_size)

     rank_array = []
     loss_array = []
     lrs = []
     etas = []
     lambs = []

     for run in range(num_runs):
          run_path = model_path+'/run'+str(run)
          Path(run_path).mkdir(parents=True, exist_ok=True)
          lr = sample_linear(lr_min,lr_max)
          eta = sample_log(eta_min,eta_max)
          lamb = sample_linear(lambda_min,lambda_max)
          lrs.append(lr)
          etas.append(eta)
          lambs.append(lamb)

          new_rank_array, new_loss_array, models, backbones = vicreg_pretraining_run(lr,weight_decay,eta,lamb,mu,batch_size,epochs,data_loaders,run,num_runs,extended_head)
          epochstrings = ['20','40','60','80','100']
          for i in range(len(models)):
            torch.save(models[i], run_path+'/'+epochstrings[i]+'model.pth')
            torch.save(backbones[i],run_path+'/'+epochstrings[i]+'backbone.pth')

          rank_array.append(new_rank_array)
          loss_array.append(new_loss_array)
     with open(path+'/lr.json', 'w') as file:
        json.dump(lrs, file)
     with open(path+'/eta.json', 'w') as file:
        json.dump(etas, file)
     with open(path+'/lambda.json', 'w') as file:
        json.dump(lambs, file)
     with open(path+'/rank.json', 'w') as file:
        json.dump(rank_array, file)
     with open(path+'/loss.json', 'w') as file:
        json.dump(loss_array, file)

     

def simclr_pretraining_routine(num_runs = 1,identifier='simclr', extended_head = False):
     path = 'pretraining/'+identifier
     model_path = path +'/models'
     Path(model_path).mkdir(parents=True, exist_ok=True)
     
     lr_min = 0.5
     lr_max = 0.9
     weight_decay_min = 1e-7
     weight_decay_max = 1e-2
     temperature_min = 0.1
     temperature_max = 0.5
     batch_size = 256 #significantly smaller dataset
     epochs = 100

     data_path = 'datasets/imagenette'
     data_loaders = get_pretrain_loaders(data_path,batch_size)

     rank_array = []
     loss_array = []
     lrs = []
     wds = []
     temps = []

     for run in range(num_runs):
          run_path = model_path+'/run'+str(run)
          Path(run_path).mkdir(parents=True, exist_ok=True)
          lr = sample_linear(lr_min,lr_max)
          weight_decay = sample_log(weight_decay_min,weight_decay_max)
          temperature = sample_linear(temperature_min,temperature_max)
          lrs.append(lr)
          wds.append(weight_decay)
          temps.append(temperature)

          new_rank_array, new_loss_array, models, backbones = simclr_pretraining_run(lr,weight_decay,temperature,batch_size,epochs,data_loaders,run,num_runs, extended_head)
          epochstrings = ['20','40','60','80','100']
          for i in range(len(models)):
            torch.save(models[i], run_path+'/'+epochstrings[i]+'model.pth')
            torch.save(backbones[i],run_path+'/'+epochstrings[i]+'backbone.pth')

          rank_array.append(new_rank_array)
          loss_array.append(new_loss_array)
     with open(path+'/lr.json', 'w') as file:
        json.dump(lrs, file)
     with open(path+'/wd.json', 'w') as file:
        json.dump(wds, file)
     with open(path+'/temp.json', 'w') as file:
        json.dump(temps, file)
     with open(path+'/rank.json', 'w') as file:
        json.dump(rank_array, file)
     with open(path+'/loss.json', 'w') as file:
        json.dump(loss_array, file)
     
def compute_acc(model,data_loader):
    model.eval()
    samples=0.0
    predicts=0.0
    with torch.no_grad():
      for inputs, labels in data_loader:
         # Assuming GPU is available, move data to GPU
         inputs, labels = inputs.to(device), labels.to(device)

         # Forward pass
         outputs = model(inputs)

         # Get predictions
         _, predicted = torch.max(outputs, 1)

         # Update counts
         samples += labels.size(0)
         predicts += (predicted == labels).sum().item()

    return predicts/samples

def classification_linear(backbone,train_loader,classes,lr,wd,epochs,run):
    in_features = 512
    linear_head = LinearHead(in_features, classes)  # num_classes is the number of classes in your dataset
    Classifier = ClassModel(backbone,linear_head)
    Classifier.to(device)

    # Crossentropy loss and Adam optimizer from pytorch
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(linear_head.parameters(), lr=lr,weight_decay=wd)
    avg_losses = []

    for epoch in range(epochs):
        total_loss = 0.0
        for batch in tqdm(train_loader, desc=f'Linear training run {run + 1}', leave=False):
            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = Classifier(inputs)

            # Compute the loss
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

        avg_loss = total_loss / len(train_loader)
        avg_losses.append(avg_loss)
    
    
    return Classifier, avg_losses

def classification_mlp(backbone,train_loader,classes,lr,wd,epochs,run):
    in_features = 512
    hidden_dims = [512,512]
    mlp_head = MLPHead(in_features, hidden_dims, classes)  # num_classes is the number of classes in your dataset
    Classifier = ClassModel(backbone,mlp_head)
    Classifier.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(mlp_head.parameters(), lr=lr,weight_decay=wd)
    avg_losses = []

    for epoch in range(epochs):
        total_loss = 0.0
        for batch in tqdm(train_loader, desc=f'MLP run {run + 1}', leave=False):
            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = Classifier(inputs)

            # Compute the loss
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

        avg_loss = total_loss / len(train_loader)
        avg_losses.append(avg_loss)
    
    
    return Classifier, avg_losses

    
def classification_routine(name,data,dino):
    data_path = 'datasets/'+data
    batch_size = 512
    if data == 'imagenette':
        train_loader, eval_loader, val_loader = get_classification_loaders(data_path,batch_size)
        classes = 10
    elif data == 'cifar10':
        train_loader, eval_loader, val_loader = get_classification_loadersCIFAR10(data_path,batch_size)
        classes = 10
    elif data == 'cifar100':
        train_loader, eval_loader, val_loader = get_classification_loadersCIFAR100(data_path,batch_size)
        classes = 100
    else:
        print('Unknown Dataset')
        return
    train_path_lin = 'classification/'+name+'/'+data+'/linear'
    store_path_lin = train_path_lin+'/models'
    train_path_mlp = 'classification/'+name+'/'+data+'/mlp'
    store_path_mlp = train_path_mlp+'/models'
    load_path = 'pretraining/'+name+'/models/run'

    Path(store_path_lin).mkdir(parents=True, exist_ok=True)
    Path(store_path_mlp).mkdir(parents=True, exist_ok=True)
    lr = 0.001
    wd = 1e-4
    epochs = 20

    train_losses_lin = []
    train_accs_lin = []
    val_accs_lin = []

    train_losses_mlp = []
    train_accs_mlp = []
    val_accs_mlp = []

    model_var = ['20','40','60','80','100']
    if dino == 1:
        model_end = '_t.pth'
    else:
        model_end = '.pth'

    run = 0
    while True:
        train_losses_lin5 = []
        train_accs_lin5 = []
        val_accs_lin5 = []
        train_losses_mlp5 = []
        train_accs_mlp5 = []
        val_accs_mlp5 = []
        model_path = load_path+str(run)+'/'+model_var[-1]+'backbone'+model_end

        try:
            
            resnet = torchvision.models.resnet18()
            backbone = nn.Sequential(*list(resnet.children())[:-1])
            backbone.load_state_dict(torch.load(model_path))
                
        except FileNotFoundError:
            print(f"No more models found. Stopping.")
            break
        except Exception as e:
            print(f"An error occurred while loading the model from {model_path}: {e}")
            break

        for i in range(5):
            model_path = load_path+str(run)+'/'+model_var[i]+'backbone'+model_end
            
            
            resnet = torchvision.models.resnet18()
            backbone = nn.Sequential(*list(resnet.children())[:-1])
            backbone.load_state_dict(torch.load(model_path))

            backbone.eval()
            for param in backbone.parameters():
                param.requires_grad = False
            

            Classifier_lin, losses_lin = classification_linear(backbone,train_loader,classes,lr,wd,epochs,run)
            Classifier_mlp, losses_mlp = classification_mlp(backbone,train_loader,classes,lr,wd,epochs,run)

            Path(store_path_lin+'/run'+str(run)).mkdir(parents=True, exist_ok=True)
            Path(store_path_mlp+'/run'+str(run)).mkdir(parents=True, exist_ok=True)

            torch.save(Classifier_lin, store_path_lin+'/run'+str(run) +'/'+model_var[i]+'model.pth')
            torch.save(Classifier_mlp, store_path_mlp+'/run'+str(run) +'/'+model_var[i]+'model.pth')

            train_losses_lin5.append(losses_lin)
            train_accs_lin5.append(compute_acc(Classifier_lin,eval_loader))
            val_accs_lin5.append(compute_acc(Classifier_lin,val_loader))
            train_losses_mlp5.append(losses_mlp)
            train_accs_mlp5.append(compute_acc(Classifier_mlp,eval_loader))
            val_accs_mlp5.append(compute_acc(Classifier_mlp,val_loader))

        train_losses_lin.append(train_losses_lin5)
        train_accs_lin.append(train_accs_lin5)
        val_accs_lin.append(val_accs_lin5)
        train_losses_mlp.append(train_losses_mlp5)
        train_accs_mlp.append(train_accs_mlp5)
        val_accs_mlp.append(val_accs_mlp5)

        # Increment run_number for the next iteration
        run += 1

    with open(train_path_lin+'/train_accs.json', 'w') as file:
        json.dump(train_accs_lin, file)
    with open(train_path_lin+'/train_losses.json', 'w') as file:
        json.dump(train_losses_lin, file)
    with open(train_path_lin+'/val_accs.json', 'w') as file:
        json.dump(val_accs_lin, file)
    with open(train_path_mlp+'/train_accs.json', 'w') as file:
        json.dump(train_accs_mlp, file)
    with open(train_path_mlp+'/train_losses.json', 'w') as file:
        json.dump(train_losses_mlp, file)
    with open(train_path_mlp+'/val_accs.json', 'w') as file:
        json.dump(val_accs_mlp, file)



def download_cifar():
    data_dir = 'datasets/cifar10'
    train_set = datasets.CIFAR10(root=data_dir, train=True, download=True)
    test_set = datasets.CIFAR10(root=data_dir, train=False, download=True)

    data_dir = 'datasets/cifar100'
    train_set = datasets.CIFAR100(root=data_dir, train=True, download=True)
    test_set = datasets.CIFAR100(root=data_dir, train=False, download=True)

def main():
    task = sys.argv[1]
    if task == 'IMAGENETTE':
        download_imagenette()
        print('Imagenette download successful.')
    if task == 'SIMCLR':
        num_runs = int(sys.argv[2])
        suffix = str(sys.argv[3])
        identifier = 'simclr'+suffix
        extended_head = bool(sys.argv[4])
        simclr_pretraining_routine(num_runs,identifier,extended_head)
        print('SIMCLR Pretraining Successful')

    if task == 'VICREG':
        num_runs = int(sys.argv[2])
        suffix = str(sys.argv[3])
        identifier = 'vicreg'+suffix
        extended_head = bool(sys.argv[4])
        vicreg_pretraining_routine(num_runs,identifier,extended_head)
        print('VICREG Pretraining Successful')

    if task == 'DINO':
        num_runs = int(sys.argv[2])
        suffix = str(sys.argv[3])
        identifier = 'dino'+suffix
        extended_head = bool(sys.argv[4])
        dino_pretraining_routine(num_runs,identifier,extended_head)
        print('DINO Pretraining Successful')

    if task == 'CLASSIFICATION':
        name = str(sys.argv[2])
        data = str(sys.argv[3])
        dino = (int(sys.argv[4]) == 1)
        classification_routine(name,data,dino)
        print('Classification Successfull')


    if task == 'CIFAR':
        download_cifar()

    if task == 'FULLCLASSIFICATION':
        name = str(sys.argv[2])
        data = ['imagenette','cifar10','cifar100']
        dino = int(sys.argv[3])
        for i in range(len(data)):
            classification_routine(name,data[i],dino)
            print(data[i]+' successfull.')
        print('Classification Successfull')




        

if __name__ == "__main__":
    raise SystemExit(main())

