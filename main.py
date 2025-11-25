import argparse
import os
from datetime import datetime

import numpy as np

from train import *

for run in range(1):  
    print(f"\n=== 开始第 {run+1} 次运行 ===")
    agents = 5
    w = np.array([[0.6, 0, 0, 0.4, 0],[0.2, 0.8, 0, 0, 0], [0.2, 0.1, 0.4, 0, 0.3], [0, 0, 0, 0.6, 0.4],[0, 0.1, 0.6, 0, 0.3]])

    dataset = "imagenet"
    # dataset = "mnist"
    epochs =20
    bs = 512   
    # lr = 0.05
    # lr_clip = 0.1
    # c_0 = 0.7

    def parse_args():
        ''' Function parses command line arguments '''
        parser = argparse.ArgumentParser()
        parser.add_argument("-t", "--test_num", default=0, type=int)
        parser.add_argument("-r", "--run_num", default=0, type=int)
        parser.add_argument("-s", "--stratified", action='store_true')
        parser.add_argument("--lr", default=0.05, type=float, help="Learning rate")
        parser.add_argument("--lr_clip", default=0.01, type=float, help="Learning rate clip")
        parser.add_argument("--c_0", default=0.7, type=float, help="C_0 value")
        return parser.parse_args()

    args = parse_args()
    cwd = os.getcwd()
    results_path = os.path.join(cwd, "results")
    if not os.path.isdir(results_path):
        os.mkdir(results_path)

    stratified = args.stratified
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    lr = args.lr
    lr_clip = args.lr_clip
    c_0 = args.c_0
    fname = os.path.join(results_path,f"{dataset}_e{epochs}_lr{lr}_lrclip{lr_clip}_c0{c_0}_hom{stratified}_{args.test_num}_{current_time}.csv")


    print(f"Test Num {args.test_num}, run num: {args.run_num}, {fname}")
    # if args.test_num == 0:
    #     DAdSGDTrainer(dataset=dataset, batch_size=bs, epochs=epochs,w=w, fname=fname, stratified=1)
    # elif args.test_num == 1:
    #     DLASTrainer(dataset=dataset, batch_size=bs, epochs=epochs, w=w, kappa=0.37, fname=fname, stratified=stratified)
    # elif args.test_num == 2:
    #     DAMSGradTrainer(dataset=dataset, batch_size=bs, epochs=epochs, w=w, fname=fname, stratified=stratified)
    # elif args.test_num == 3:
    #     DAdaGradTrainer(dataset=dataset, batch_size=bs, epochs=epochs, w=w, fname=fname, stratified=stratified)
    # elif args.test_num == 4:
    #     CDSGDTrainer(dataset=dataset, batch_size=bs, epochs=epochs, num=0.001, w=w, fname=fname, stratified=stratified)
    # elif args.test_num == 5:
    #     CDSGDPTrainer(dataset=dataset, batch_size=bs, epochs=epochs, num=0.001, w=w, fname=fname, stratified=stratified)
    # elif args.test_num == 6:
    #     CDSGDNTrainer(dataset=dataset, batch_size=bs, epochs=epochs, num=0.001, w=w, fname=fname, stratified=stratified)
    # DingtieTrainer (dataset=dataset, batch_size=bs, epochs=epochs,w=w, lr=lr, fname=fname, stratified=1)
    if args.test_num == 0:
        trainer_name = "DAdSGDTrainer"
        trainer = DAdSGDTrainer(dataset=dataset, batch_size=bs, epochs=epochs, w=w, fname=fname, stratified=1)
    elif args.test_num == 1:
        trainer_name = "DLASTrainer"
        trainer = DLASTrainer(dataset=dataset, batch_size=bs, epochs=epochs, w=w, kappa=0.37, fname=fname, stratified=stratified)
    elif args.test_num == 2:
        trainer_name = "DAMSGradTrainer"
        trainer = DAMSGradTrainer(dataset=dataset, batch_size=bs, epochs=epochs, w=w, fname=fname, stratified=stratified)
    elif args.test_num == 3:
        trainer_name = "DAdaGradTrainer"
        trainer = DAdaGradTrainer(dataset=dataset, batch_size=bs, epochs=epochs, w=w, fname=fname, stratified=stratified)
    elif args.test_num == 4:
        trainer_name = "CDSGDTrainer"
        trainer = CDSGDTrainer(dataset=dataset, batch_size=bs, epochs=epochs, num=lr, w=w, fname=fname, stratified=stratified)
    elif args.test_num == 5:
        trainer_name = "CDSGDPTrainer"
        trainer = CDSGDPTrainer(dataset=dataset, batch_size=bs, epochs=epochs, num=0.001, w=w, fname=fname, stratified=stratified)
    elif args.test_num == 6:
        trainer_name = "CDSGDNTrainer"
        trainer = CDSGDNTrainer(dataset=dataset, batch_size=bs, epochs=epochs, num=0.001, w=w, fname=fname, stratified=stratified)
    elif args.test_num == 7:
        trainer_name = "CDSGTrainer"
        trainer = DingtieTrainer(dataset=dataset, batch_size=bs, epochs=epochs, w=w, lr=lr, fname=fname, stratified=1)
    else:
        trainer_name = "DCGTTrainer"
        trainer = DCGTTrainer(dataset=dataset, batch_size=bs, epochs=epochs, w=w, c_0=c_0, lr=lr_clip, fname=fname, stratified=1)


    