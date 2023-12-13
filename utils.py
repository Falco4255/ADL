import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


### From: Emerging Properties in Self-Supervised Vision Transformers https://arxiv.org/abs/2104.14294 
class DINOLoss(nn.Module):
    def __init__(self, tpt=1.0,tps=1.0,m=0.9,out_dim=8192):
        super(DINOLoss, self).__init__()
        self.tpt = tpt
        self.tps = tps
        self.C = torch.zeros(out_dim, dtype=torch.float).to(device)
        self.m = m
        self.eps = torch.tensor(1e-10, dtype=torch.float).to(device)

    def HDino(self,t, s):
        t = t.detach() # stop gradient
        s = F.softmax(s / self.tps, dim=1)
        s = torch.add(s,self.eps)
        t = F.softmax((t - self.C) / self.tpt, dim=1) # center + sharpen
        return - (t * torch.log(s)).sum(dim=1).mean()

    def forward(self, output_t, output_s):
        loss = torch.tensor(0.0, dtype=torch.float).to(device)
        for i in range(len(output_t)):
            for j in range(len(output_s)):
                if i==j:
                    continue
                loss += self.HDino(output_t[i],output_s[j])
                
        self.C = self.m*self.C + (1-self.m)*torch.mean(torch.stack([output.mean(dim=0) for output in output_t]),dim=0)

        return loss


#After the documentation in RankME https://arxiv.org/abs/2210.02885
#returns entropic rank and robust rank
def RankME(model,dataloader):
    # Pass the entire dataset through the model
    all_outputs = []
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch[0]
            inputs = inputs.to(device)
            outputs = model(inputs)
            all_outputs.append(outputs)

    # Concatenate the outputs to get the final result
    all_outputs = torch.cat(all_outputs, dim=0).to("cpu")

    max_dim = np.max([all_outputs.shape[0],all_outputs.shape[1]])
    eps = 1e-7

    # Perform Singular Value Decomposition
    U, S, V = np.linalg.svd(np.array(all_outputs))

    sigma_max = S[0]
    threshold = sigma_max*max_dim*eps

    if len(S) == max_dim:
        sys.exit('Less than 512 images, somethin went wrong while computing RankME')

    s_norm = np.linalg.norm(S,1)
    p = S/s_norm + eps
    rank = np.sum((S>=threshold))
    entropy = -np.sum(p*np.log(p))


    return [np.exp(entropy).item(), rank.item()]

def sample_linear(min,max):
     rand = random.uniform(min,max)
     return rand

def sample_log(min,max):
     rand = random.uniform(np.log(min),np.log(max))
     return np.exp(rand)


class CosineSchedule(nn.Module):
    def __init__(self, min=1.0,max=1.0,tot_epochs=100):
        super(CosineSchedule, self).__init__()
        self.min = min
        self.max = max
        self.epochs = tot_epochs
        self.epoch = torch.tensor(0, dtype=torch.float)

    def step(self):
        self.epoch = self.epoch + torch.tensor(1, dtype=torch.float).to(device)
        cos_ann = 0.5 * (1 + torch.cos(self.epoch / self.epochs * torch.pi))
        return self.min + (self.max - self.min) * cos_ann