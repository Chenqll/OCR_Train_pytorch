from __future__ import print_function
import argparse
import random
import torch
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import numpy as np
from torch.nn import CTCLoss
import utils
import mydataset
import crnn as crnn
import config
from online_test import val_model
from config import parseArgs

config.imgW = 800
config.alphabet = config.alphabet_v2
config.nclass = len(config.alphabet) + 1
config.saved_model_prefix = "CRNN"
config.keep_ratio = True
config.use_log = True
config.batchSize = 1
config.workers = 0
config.adam = True
import os
import datetime

class Trainer:
    def __init__(self,args):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        config.manualSeed = random.randint(1, 10000)
        print("Random Seed: ", config.manualSeed)
        random.seed(config.manualSeed)
        np.random.seed(config.manualSeed)
        torch.manual_seed(config.manualSeed)

        train_dataset = mydataset.MyDataset(info_filename=args.train_infofile)
        assert train_dataset
        if not config.random_sample:
            sampler = mydataset.randomSequentialSampler(train_dataset, args.batchSize)
        else:
            sampler = None
        self.train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batchSize,
            shuffle=True,
            sampler=sampler,
            num_workers=int(args.workers),
            collate_fn=mydataset.alignCollate(
                imgH=config.imgH, imgW=config.imgW, keep_ratio=config.keep_ratio
            ),
        )

        self.test_dataset = mydataset.MyDataset(
            info_filename=args.val_infofile,
            transform=mydataset.resizeNormalize((config.imgW, config.imgH), is_test=True),
        )
        self.converter = utils.strLabelConverter(config.alphabet)
        self.criterion = CTCLoss(reduction="sum", zero_infinity=True)
        self.best_acc = 0.9

        self.crnn = crnn.CRNN(config.imgH, config.nc, config.nclass, config.nh)
        if args.pretrained_model != '' and os.path.exists(args.pretrained_model):
            print('loading pretrained model from %s' % args.pretrained_model)
            self.crnn.load_state_dict(torch.load(args.pretrained_model, map_location=torch.device('cpu')))
        else:
            self.crnn.apply(self.weights_init)

        self.device = torch.device('cpu')
        if config.cuda:
            self.crnn.cuda()
            # crnn = torch.nn.DataParallel(crnn, device_ids=range(opt.ngpu))
            # image = image.cuda()
            self.device = torch.device('cuda:0')
            self.criterion = self.criterion.cuda()

        self.loss_avg = utils.averager()

        # setup optimizer
        if config.adam:
            self.optimizer = optim.Adam(self.crnn.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
        elif config.adadelta:
            self.optimizer = optim.Adadelta(self.crnn.parameters(), lr=args.lr)
        else:
            self.optimizer = optim.RMSprop(self.crnn.parameters(), lr=args.lr)

            self.train_val(self,self.crnn, self.test_dataset, self.criterion)


    def train_val(self):
        for epoch in range(args.niter):
            self.loss_avg.reset()
            print("epoch {}....".format(epoch))
            self.train_iter = iter(self.train_loader)
            i = 0
            n_batch = len(self.train_loader)
            while i < len(self.train_loader):
                for p in self.crnn.parameters():
                    p.requires_grad = True
                self.crnn.train()
                cost = self.trainBatch()
                print(
                    "epoch: {} iter: {}/{} Train loss: {:.3f}".format(
                        epoch, i, n_batch, cost.item()
                    )
                )
                self.loss_avg.add(cost)
                self.loss_avg.add(cost)
                i += 1
            print("Train loss: %f" % (self.loss_avg.val()))

        print("Start val")
        for p in self.crnn.parameters():
            p.requires_grad = False

        num_correct, num_all = val_model(
            args.val_infofile,
            self.crnn,
            True,
        )
        accuracy = num_correct / num_all

        print("ocr_acc: %f" % (accuracy))
        global best_acc
        if accuracy > self.best_acc:
            self.best_acc = accuracy
            torch.save(
                self.crnn.state_dict(),
                "{}/{}_{}_{}.pth".format(
                    args.saved_model_dir,
                    args.saved_model_prefix,
                    epoch,
                    int(self.best_acc * 1000),
                ),
            )
        torch.save(
            self.crnn.state_dict(),
            "{}/{}.pth".format(args.saved_model_dir, args.saved_model_prefix),
        )

    def trainBatch(self):
        data = self.train_iter.next()
        cpu_images, cpu_texts = data
        batch_size = cpu_images.size(0)
        image = cpu_images.to(self.device)

        text, length = self.converter.encode(cpu_texts)

        preds = self.crnn(image)
        preds_size = Variable(
            torch.IntTensor([preds.size(0)] * batch_size)
        )
        cost = self.criterion(preds.log_softmax(2).cpu(), text, preds_size, length) / batch_size
        if torch.isnan(cost):
            print(batch_size, cpu_texts)
        else:
            self.crnn.zero_grad()
            cost.backward()
            self.optimizer.step()
        return cost

if __name__ == "__main__":
    args = parseArgs()
    trainer = Trainer(args)
    trainer.train_val()