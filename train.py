from collections import OrderedDict
import copy
import csv
import random
from random import sample, shuffle
from time import perf_counter
import warnings
import math
import time
import os

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from models import *
from ops import Dingtie, DCGT, CDSGD
import torch.optim as optim
import h5py
warnings.filterwarnings("ignore")
import wandb
from torch.utils.data import Dataset
from PIL import Image

# Enable cuDNN benchmarking for optimized performance
torch.backends.cudnn.benchmark = True
# Optionally enable the fastest cuDNN algorithms
torch.backends.cudnn.fastest = True


class DTrainer:
    def __init__(self,
                 dataset="cifar10",
                 batch_size=40,
                 epochs=6,
                 num=0.5,
                 w=None,
                 w2=None,
                 w3=None,
                 fname=None,
                 stratified=True,
                 lr=1,
                 c_0=0.1,
                 workers=4,
                 agents=5,
                 kmult=0.0,
                 exp=0.7,
                 kappa=0.9):

    
        self.wandb = wandb
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_accuracy = []
        self.test_accuracy = []
        self.train_iterations = []
        self.test_iterations = []
        self.lr_logs = {}
        self.lambda_logs = {}
        self.loss_list = []
        self.train_loader = {}
        self.test_loader = {}
        # Initialize wandb project
        self.wandb.init(project="imagenet_pretrained_mobilenetv3", config={
            "dataset": dataset,
            "batch_size": batch_size,
            "epochs": epochs,
            "lr": lr,
            "c_0": c_0,
            "workers": workers,
            "agents": agents,
            "kmult": kmult,
            "exp": exp,
            "kappa": kappa
        })

        self.dataset = dataset
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.c_0 = c_0
        self.workers = workers
        self.agents = agents
        self.num = num
        self.class_num = 10 
        self.kmult = kmult
        self.exp = exp
        self.kappa = kappa
        self.fname = fname
        self.stratified = stratified
        self.load_data()
        self.w = w
        self.w2 = w2
        self.w3 = w3
        self.criterion = torch.nn.CrossEntropyLoss()
        self.agent_setup()

    def _log(self, accuracy):
        ''' Helper function to log accuracy values'''
        self.train_accuracy.append(accuracy)
        self.train_iterations.append(self.running_iteration)

    def _save(self):
        with open(self.fname, mode='a') as csv_file:
            file = csv.writer(csv_file, lineterminator = '\n')
            file.writerow([f"{self.opt_name}, {self.lr}, {self.c_0}, {self.batch_size}, {self.epochs}"])
            file.writerow(self.train_iterations)
            file.writerow(self.train_accuracy)
            file.writerow(self.test_iterations)
            file.writerow(self.test_accuracy)
            file.writerow(self.loss_list)
            file.writerow(["ETA"])
            for i in range(self.agents):
                file.writerow(self.lr_logs[i])
            if self.opt_name == "DLAS":
                file.writerow(["LAMBDA"])
                for i in range(self.agents):
                    file.writerow(self.lambda_logs[i])
            file.writerow([])



    def load_data(self):
        print("==> Loading Data")

        if self.dataset == 'cifar10':
            transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])

            transform_test = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
            self.class_num = 10
            trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
            testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
            
        elif self.dataset == "mnist":
            transform_train = transforms.Compose([transforms.ToTensor(),])
            transform_test = transforms.Compose([transforms.ToTensor(),])

            self.class_num = 10
            trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_train)
            testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_test)
        
        elif self.dataset == "imagenet":
            # Load from H5 files for faster training
            h5_file_train = '/home/yanan/Yanan/imagenet_h5/imagenet_train.h5'
            h5_file_val = '/home/yanan/Yanan/imagenet_h5/imagenet_val.h5'
            
            # Define transforms for ImageNet
            transform_train = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]),
            ])

            transform_val = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]),
            ])
            
            if os.path.exists(h5_file_train) and os.path.exists(h5_file_val):
                print("Loading ImageNet from H5 files...")
                trainset = H5Dataset(h5_file_train, transform=transform_train)
                testset =H5Dataset(h5_file_val, transform=transform_val)
            else:
                print(f"H5 files not found. Expected at:")
                print(f"  {h5_file_train}")
                print(f"  {h5_file_val}")
                raise FileNotFoundError("ImageNet H5 files not found. Please run convert_imagenet_to_h5.py first.")

        else:
            raise ValueError(f'{self.dataset} is not supported')
            
        if self.dataset == "imagenet":
            self.stratified = True

        if self.stratified:
            train_len, test_len = int(len(trainset)), int(len(testset))

            temp_train = torch.utils.data.random_split(trainset, [int(train_len//self.agents)]*self.agents)
            
            for i in range(self.agents):
                self.train_loader[i] = torch.utils.data.DataLoader(temp_train[i], batch_size=self.batch_size, shuffle=True, num_workers=8, pin_memory=True)

            self.test_loader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=8, pin_memory=True)
        else:
            train_len, test_len = int(len(trainset)), int(len(testset))
            idxs = {}
            for i in range(0, 10, 2):
                arr = np.array(trainset.targets, dtype=int)
                idxs[int(i/2)] = list(np.where(arr == i)[0]) + list(np.where(arr == i+1)[0])
                shuffle(idxs[int(i/2)])
            
            percent_main = 0.5
            percent_else = (1 - percent_main) / (self.agents-1)
            main_samp_num = int(percent_main * len(idxs[0]))
            sec_samp_num = int(percent_else * len(idxs[0]))

            for i in range(self.agents):
                agent_idxs = []
                for j in range(self.agents):
                    if i == j:
                        agent_idxs.extend(sample(idxs[j], main_samp_num))
                    else:
                        agent_idxs.extend(sample(idxs[j], sec_samp_num))
                    idxs[j] = list(filter(lambda x: x not in agent_idxs, idxs[j]))
                temp_train = copy.deepcopy(trainset)
                temp_train.targets = [temp_train.targets[i] for i in agent_idxs]
                temp_train.data = [temp_train.data[i] for i in agent_idxs]
                self.train_loader[i] = torch.utils.data.DataLoader(temp_train, batch_size=self.batch_size, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True, prefetch_factor=4)
            self.test_loader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=8, pin_memory=True, persistent_workers=True, prefetch_factor=4)


    def agent_setup(self):
        for i in range(self.agents):
            self.lr_logs[i] = []
            self.lambda_logs[i] = []

        self.agent_models = {}
        self.agent_optimizers = {}

        if self.dataset == 'cifar10':
            model = CifarCNN()
        
        elif self.dataset == "imagenet":
            model = ImageNetModel(num_classes=1000)

        elif self.dataset == "mnist":
            model = MnistCNN()

        for i in range(self.agents):
            if i == 0:
                if int(torch.cuda.device_count()) > 1:
                    self.agent_models[i] = torch.nn.DataParallel(model)
                else:
                    self.agent_models[i] = model
            else:
                if int(torch.cuda.device_count()) > 1:
                    self.agent_models[i] = copy.deepcopy(self.agent_models[0])
                else:
                    self.agent_models[i] = copy.deepcopy(model)

            self.agent_models[i].to(self.device)
            self.agent_models[i].train()


            self.agent_optimizers[i] = self.opt(
                params=self.agent_models[i].parameters(),
                idx=i,
                w=self.w,
                agents=self.agents,
                lr=self.lr,
                c_0=self.c_0,
                num=self.num,
                kmult=self.kmult,
                device=self.device,
                kappa=self.kappa,
                stratified=self.stratified
            )

    def eval(self, dataloader):
        total_acc, total_count = 0, 0

        with torch.no_grad():

            for i in range(self.agents):
                self.agent_models[i].eval()

                for inputs, labels in dataloader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    predicted_label = self.agent_models[i](inputs)

                    total_acc += (predicted_label.argmax(1) == labels).sum().item()
                    total_count += labels.size(0)

        self.test_iterations.append(self.running_iteration)
        self.test_accuracy.append(total_acc / total_count)

        return total_acc / total_count

    def it_logger(self, total_acc, total_count, epoch, log_interval, tot_loss):
        self._log(total_acc / total_count)
        t_acc = self.eval(self.test_loader)
        for i in range(self.agents):
            self.lr_logs[i].append(self.agent_optimizers[i].collect_params(lr=True))

        train_acc = total_acc / total_count
        test_acc = t_acc
        loss_val = tot_loss / (self.agents * log_interval)
        print(
            f"Epoch: {epoch + 1}, Iteration: {self.running_iteration}, " +
            f"Accuracy: {train_acc:.4f}, " +
            f"Test Accuracy: {test_acc:.4f}, " +
            f"Loss: {loss_val:.4f}, "
        )

        self.loss_list.append(loss_val)
        # Log metrics to wandb
        self.wandb.log({
            "train_accuracy": train_acc,
            "test_accuracy": test_acc,
            "loss": loss_val,
            "epoch": epoch + 1,
            "iteration": self.running_iteration
        })

    def trainer(self):
        print(
            f"==> Starting Training for {self.opt_name}, {self.epochs} epochs and {self.agents} agents on the {self.dataset} dataset, via {self.device}")

        for i in range(self.epochs):
            start_time = time.time()  # Start timing
            self.epoch_iterations(i, self.train_loader)
            end_time = time.time()  # End timing
            elapsed_time = end_time - start_time
            print(f"Epoch {i + 1}/{self.epochs} completed in {elapsed_time:.2f} seconds")

    def validate(self, dataloader):
        self.agent_models[0].eval()
        top1, top5, count = 0, 0, 0
        with torch.no_grad(), torch.cuda.amp.autocast():
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.agent_models[0](inputs)
                _, pred = outputs.topk(5, 1, True, True)
                correct = pred.eq(labels.view(-1, 1).expand_as(pred))
                top1 += correct[:, :1].sum().item()
                top5 += correct.sum().item()
                count += labels.size(0)
        print(f"Validation Top-1: {100 * top1 / count:.2f}% | Top-5: {100 * top5 / count:.2f}%")


class H5Dataset(Dataset):
    """Optimized HDF5 loader for multi-worker ImageNet training"""

    def __init__(self, h5_file_path, transform=None):
        self.h5_file_path = h5_file_path
        self.transform = transform
        self.file = None
        self.img_ds = None
        self.lbl_ds = None

        with h5py.File(self.h5_file_path, "r") as f:
            self.length = f["images"].shape[0]

    def _lazy_open(self):
        if self.file is None:
            # SWMR = safe for multi-worker reading
            self.file = h5py.File(self.h5_file_path, "r", swmr=True, libver="latest")
            self.img_ds = self.file["images"]
            self.lbl_ds = self.file["labels"]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        self._lazy_open()

        img = self.img_ds[idx]   # numpy uint8
        label = int(self.lbl_ds[idx])

        img = Image.fromarray(img)

        if self.transform:
            img = self.transform(img)

        return img, label


class DingtieTrainer(DTrainer):
    def __init__(self, *args, **kwargs):
        self.opt = Dingtie
        self.opt_name = "Dingtie"
        super().__init__(*args, **kwargs)
        self.trainer()
        self._save()

    def epoch_iterations(self, epoch,
                         dataloader):

        if self.dataset == "imagenet":
            log_interval = 100   
        elif self.dataset == "cifar10":
            log_interval = int(len(dataloader[0]) - 1)
        else:
            log_interval = 25

        start_time = perf_counter()
        loss = {}
        total_acc, total_count, tot_loss = 0, 0, 0

        for idx, data in enumerate(zip(*dataloader.values())):
            self.running_iteration = idx + epoch * len(dataloader[0])
            vars, grads, s, z, prev_grad = {}, {}, {}, {}, {}

            for i in range(self.agents):
                inputs, labels = data[i]
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                self.agent_optimizers[i].zero_grad()
                predicted_label = self.agent_models[i](inputs)
                loss[i] = self.criterion(predicted_label, labels)
                loss[i].backward()
                vars[i], grads[i] = self.agent_optimizers[i].collect_params()

                if self.running_iteration == 0:
                    s[i] = grads[i]
                    prev_grad[i] = grads[i]
                else:
                    s[i] = self.agent_optimizers[i].collect_s()
                    prev_grad[i] = self.agent_optimizers[i].collect_prev_grad()

                total_acc += (predicted_label.argmax(1) == labels).sum().item()
                total_count += labels.size(0)
                tot_loss += loss[i].item()

            for i in range(self.agents):
                self.agent_optimizers[i].step(self.running_iteration, vars=vars, grads=grads, s_all=s,
                                              prev_grad_all=prev_grad, ns='yes')

            if idx % log_interval == 0 and idx > 0 and epoch % 1 == 0:
                elapsed_time = perf_counter() - start_time
                print(f"Elapsed Time: {elapsed_time:.2f} seconds")
                self.it_logger(total_acc, total_count, epoch, log_interval, tot_loss)
                total_acc, total_count, tot_loss = 0, 0, 0
                for i in range(self.agents):
                    self.agent_models[i].train()

        return total_acc



class DCGTTrainer(DTrainer):
    def __init__(self, *args, **kwargs):
        self.opt = DCGT
        self.opt_name = "DCGT"
        super().__init__(*args, **kwargs)
        self.trainer()
        self._save()

    def epoch_iterations(self, epoch,
                         dataloader):

        if self.dataset == "imagenet":
            log_interval = 100   
        elif self.dataset == "cifar10":
            log_interval = int(len(dataloader[0]) - 1)
        else:
            log_interval = 25

        start_time = perf_counter()
        loss = {}
        total_acc, total_count, tot_loss = 0, 0, 0

        for idx, data in enumerate(zip(*dataloader.values())):
            self.running_iteration = idx + epoch * len(dataloader[0])
            vars, grads, s, z, prev_grad = {}, {}, {}, {}, {}

            for i in range(self.agents):
                inputs, labels = data[i]
                inputs, labels = inputs.to (self.device), labels.to(self.device)

                self.agent_optimizers[i].zero_grad()
                predicted_label = self.agent_models[i](inputs)
                loss[i] = self.criterion(predicted_label, labels)
                loss[i].backward()
                vars[i], grads[i] = self.agent_optimizers[i].collect_params()

                if self.running_iteration == 0:
                    s[i] = grads[i]
                    prev_grad[i] = grads[i]
                else:
                    s[i] = self.agent_optimizers[i].collect_s()
                    prev_grad[i] = self.agent_optimizers[i].collect_prev_grad()

                total_acc += (predicted_label.argmax(1) == labels).sum().item()
                total_count += labels.size(0)
                tot_loss += loss[i].item()

            for i in range(self.agents):
                self.agent_optimizers[i].step(self.running_iteration, vars=vars, grads=grads, s_all=s,
                                              prev_grad_all=prev_grad)


            if idx % log_interval == 0 and idx > 0 and epoch % 1 == 0:
                elapsed_time = perf_counter() - start_time
                print(f"Elapsed Time: {elapsed_time:.2f} seconds")
                self.it_logger(total_acc, total_count, epoch, log_interval, tot_loss)
                total_acc, total_count, tot_loss = 0, 0, 0
                for i in range(self.agents):
                    self.agent_models[i].train()

        return total_acc

class CDSGDTrainer(DTrainer):
    def __init__(self, *args, **kwargs):
        self.opt = CDSGD
        self.opt_name="CDSGD"
        super().__init__(*args, **kwargs)
        self.trainer()
        self._save()

    def epoch_iterations(self, epoch, dataloader):
        
        start_time = perf_counter()
        if self.dataset == "imagenet":
            log_interval = 100   # print every ~100 steps
        elif self.dataset == "cifar10":
            log_interval = int(len(dataloader[0]) - 1)
        else:
            log_interval = 25

        
        loss = {}
        total_acc, total_count, tot_loss = 0, 0, 0

        for idx, data in enumerate(zip(*dataloader.values())):
            self.running_iteration = idx + epoch * len(dataloader[0])
            vars, grads = {}, {}

            for i in range(self.agents):
                self.agent_optimizers[i].zero_grad()
                inputs, labels = data[i]
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                predicted_label = self.agent_models[i](inputs)
                loss[i] = self.criterion(predicted_label, labels)
                loss[i].backward()
                vars[i], grads[i] = self.agent_optimizers[i].collect_params()


                total_acc += (predicted_label.argmax(1) == labels).sum().item()
                total_count += labels.size(0)

                tot_loss += loss[i].item()
            
            for i in range(self.agents):
                self.agent_optimizers[i].step(self.running_iteration, vars=vars)
            
            if idx % log_interval == 0 and idx > 0 and epoch % 1 == 0:
                elapsed_time = perf_counter() - start_time
                print(f"Elapsed Time: {elapsed_time:.2f} seconds")
                self.it_logger(total_acc, total_count, epoch, log_interval, tot_loss)
                total_acc, total_count, tot_loss = 0, 0, 0
                for i in range(self.agents):
                    self.agent_models[i].train()

        return total_acc
