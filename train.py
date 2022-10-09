import os
import glob
import torch
import torchsummary
import re
import random
import numpy as np
import collections
from itertools import product
from argparse import ArgumentParser
import traceback
import sys

from retinanet import model
from retinanet.dataloader import BeatDataset, collater
from retinanet.dstcn import dsTCNModel

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

torch.multiprocessing.set_sharing_strategy('file_system')

torch.backends.cudnn.benchmark = True

parser = ArgumentParser()

# add PROGRAM level args
parser.add_argument('--dataset', type=str, default='ballroom')
parser.add_argument('--beatles_audio_dir', type=str, default='./data')
parser.add_argument('--beatles_annot_dir', type=str, default='./data')
parser.add_argument('--ballroom_audio_dir', type=str, default='./data')
parser.add_argument('--ballroom_annot_dir', type=str, default='./data')
parser.add_argument('--hainsworth_audio_dir', type=str, default='./data')
parser.add_argument('--hainsworth_annot_dir', type=str, default='./data')
parser.add_argument('--rwc_popular_audio_dir', type=str, default='./data')
parser.add_argument('--rwc_popular_annot_dir', type=str, default='./data')
parser.add_argument('--preload', action="store_true")
parser.add_argument('--audio_sample_rate', type=int, default=44100)
parser.add_argument('--target_factor', type=int, default=256) # block 하나당 곱하기 2
parser.add_argument('--shuffle', type=bool, default=True)
parser.add_argument('--train_subset', type=str, default='train')
parser.add_argument('--val_subset', type=str, default='val')
parser.add_argument('--train_length', type=int, default=65536)
parser.add_argument('--train_fraction', type=float, default=1.0)
parser.add_argument('--eval_length', type=int, default=131072)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--num_workers', type=int, default=0)
parser.add_argument('--augment', action='store_true')
parser.add_argument('--dry_run', action='store_true')
parser.add_argument('--depth', default=50)
parser.add_argument('--epochs', help='Number of epochs', type=int, default=100)
#parser.add_argument('--lr', type=float, default=1e-2)
parser.add_argument('--patience', type=int, default=40)
# --- tcn model related ---
parser.add_argument('--ninputs', type=int, default=1)
parser.add_argument('--noutputs', type=int, default=2)
parser.add_argument('--nblocks', type=int, default=8)
parser.add_argument('--kernel_size', type=int, default=15)
parser.add_argument('--stride', type=int, default=2)
parser.add_argument('--dilation_growth', type=int, default=8)
parser.add_argument('--channel_growth', type=int, default=1)
parser.add_argument('--channel_width', type=int, default=32)
parser.add_argument('--stack_size', type=int, default=4)
parser.add_argument('--grouped', default=False, action='store_true')
parser.add_argument('--causal', default=False, action="store_true")
parser.add_argument('--skip_connections', default=False, action="store_true")
parser.add_argument('--norm_type', type=str, default='BatchNorm')
parser.add_argument('--act_type', type=str, default='PReLU')
parser.add_argument('--fcos', action='store_true')

# THIS LINE IS KEY TO PULL THE MODEL NAME
temp_args, _ = parser.parse_known_args()

# parse them args
args = parser.parse_args()

datasets = ["ballroom"]

# set the seed
seed = 42

random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

#
args.default_root_dir = os.path.join("lightning_logs", "full")
print(args.default_root_dir)

state_dicts = glob.glob('./checkpoints/*.ckpt')
start_epoch = 0
checkpoint_path = None
if len(state_dicts) > 0:
    checkpoint_path = state_dicts[-1]
    start_epoch = int(re.search("epoch=(.*)-step", checkpoint_path).group(1)) + 1
    print("loaded:" + checkpoint_path)
else:
    print("no checkpoint found")

# setup the dataloaders
train_datasets = []
val_datasets = []

for dataset in datasets:
    if dataset == "beatles":
        audio_dir = args.beatles_audio_dir
        annot_dir = args.beatles_annot_dir
    elif dataset == "ballroom":
        audio_dir = args.ballroom_audio_dir
        annot_dir = args.ballroom_annot_dir
    elif dataset == "hainsworth":
        audio_dir = args.hainsworth_audio_dir
        annot_dir = args.hainsworth_annot_dir
    elif dataset == "rwc_popular":
        audio_dir = args.rwc_popular_audio_dir
        annot_dir = args.rwc_popular_annot_dir

    train_dataset = BeatDataset(audio_dir,
                                    annot_dir,
                                    dataset=dataset,
                                    audio_sample_rate=args.audio_sample_rate,
                                    target_factor=args.target_factor,
                                    subset="train",
                                    fraction=args.train_fraction,
                                    augment=args.augment,
                                    half=True,
                                    preload=args.preload,
                                    length=args.train_length,
                                    dry_run=args.dry_run)
    train_datasets.append(train_dataset)

    val_dataset = BeatDataset(audio_dir,
                                 annot_dir,
                                 dataset=dataset,
                                 audio_sample_rate=args.audio_sample_rate,
                                 target_factor=args.target_factor,
                                 subset="val",
                                 augment=False,
                                 half=True,
                                 preload=args.preload,
                                 length=args.eval_length,
                                 dry_run=args.dry_run)
    val_datasets.append(val_dataset)

train_dataset_list = torch.utils.data.ConcatDataset(train_datasets)
val_dataset_list = torch.utils.data.ConcatDataset(val_datasets)

train_dataloader = torch.utils.data.DataLoader(train_dataset_list, 
                                                shuffle=args.shuffle,
                                                batch_size=args.batch_size,
                                                num_workers=args.num_workers,
                                                pin_memory=True,
                                                collate_fn=collater)
val_dataloader = torch.utils.data.DataLoader(val_dataset_list, 
                                            shuffle=args.shuffle,
                                            batch_size=1,
                                            num_workers=args.num_workers,
                                            pin_memory=False,
                                            collate_fn=collater)

dict_args = vars(args)

if __name__ == '__main__':
    # Create the model
    if args.depth == 18:
        retinanet = model.resnet18(**dict_args)
    elif args.depth == 34:
        retinanet = model.resnet34(**dict_args)
    elif args.depth == 50:
        retinanet = model.resnet50(**dict_args)
    elif args.depth == 101:
        retinanet = model.resnet101(**dict_args)
    elif args.depth == 152:
        retinanet = model.resnet152(**dict_args)
    else:
        raise ValueError('Unsupported model depth, must be one of 18, 34, 50, 101, 152')

    use_gpu = True

    if use_gpu:
        if torch.cuda.is_available():
            retinanet = retinanet.cuda()

    if torch.cuda.is_available():
        retinanet = torch.nn.DataParallel(retinanet).cuda()
    else:
        retinanet = torch.nn.DataParallel(retinanet)

    optimizer = torch.optim.Adam(retinanet.parameters(), lr=1e-5)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)

    loss_hist = collections.deque(maxlen=500)

    retinanet.train()
    retinanet.module.freeze_bn()

    print('Num training images: {}'.format(len(train_dataset_list)))

    for epoch_num in range(args.epochs):
        retinanet.train()
        retinanet.module.freeze_bn()

        epoch_loss = []

        for iter_num, data in enumerate(train_dataloader):
            audio, target = data

            try:
                optimizer.zero_grad()

                if args.fcos:
                    classification_loss, regression_loss, centerness_loss = retinanet(data)
                else:
                    classification_loss, regression_loss = retinanet(data)
                    centerness_loss = torch.zeros(1)
    
                classification_loss = classification_loss.mean()
                regression_loss = regression_loss.mean()
                centerness_loss = centerness_loss.mean()

                loss = classification_loss + regression_loss + centerness_loss

                if bool(loss == 0):
                    continue

                loss.backward()

                torch.nn.utils.clip_grad_norm_(retinanet.parameters(), 0.1)

                optimizer.step()

                loss_hist.append(float(loss))

                epoch_loss.append(float(loss))

                print(
                    'Epoch: {} | Iteration: {} | Classification loss: {:1.5f} | Regression loss: {:1.5f} | Centerness loss: {:1.5f} | Running loss: {:1.5f}'.format(
                        epoch_num, iter_num, float(classification_loss), float(regression_loss), float(centerness_loss), np.mean(loss_hist)))

                del classification_loss
                del regression_loss
                del centerness_loss
            except KeyboardInterrupt:
                sys.exit()
            except Exception as e:
                print(e)
                traceback.print_exc()
                continue

        scheduler.step(np.mean(epoch_loss))

        #torch.save(retinanet.module, '{}_retinanet_{}.pt'.format(parser.dataset, epoch_num))

    retinanet.eval()

    torch.save(retinanet, 'model_final.pt')
