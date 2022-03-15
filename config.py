import argparse

import keys
alphabet = keys.alphabet
alphabet_v2 = keys.alphabet_v2
nclass = len(alphabet) + 1
cuda = False
ngpu = 1
use_log = False
remove_blank = False
imgH = 32
imgW = 280
nc = 1
nh = 256
experiment = None
displayInterval = 500
n_test_disp = 10
valInterval = 500
saveInterval = 500
adam = False
adadelta = False
keep_ratio = False
random_sample = True

def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_infofile",type=str,default="./train_data/train.txt")
    parser.add_argument("--train_infofile_fullimg", type=str, default="")
    parser.add_argument("--val_infofile", type=str, default="./train_data/train.txt")

    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batchSize", type=int, default=50)

    parser.add_argument("--niter", type=int, default=10)
    parser.add_argument("--lr", type=int, default=0.0003)
    parser.add_argument("--beta1", type=int, default=0.5)

    parser.add_argument("--pretrained_model", type=str, default="E:\ocr_train_local\checkpoints\CRNN.pth")
    parser.add_argument("--saved_model_dir", type=str, default="crnn_models")
    parser.add_argument("--saved_model_prefix", type=str, default="CRNN")

    args = parser.parse_args()
    return args