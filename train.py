import os
import glob
import torch
import torchsummary
import re
from itertools import product
import pytorch_lightning as pl
from argparse import ArgumentParser

from pytorch_lightning.callbacks import ModelCheckpoint
from retinanet import model
from retinanet.dataloader import BeatDataset

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

checkpoint_callback = ModelCheckpoint(
    verbose=True,
    monitor='val_loss/Joint F-measure',
    mode='max'
)

# add all the available trainer options to argparse
parser = pl.Trainer.add_argparse_args(parser)

# THIS LINE IS KEY TO PULL THE MODEL NAME
temp_args, _ = parser.parse_known_args()

# let the model add what it wants
parser = dsTCNModel.add_model_specific_args(parser)

# parse them args
args = parser.parse_args()

datasets = ["ballroom", "hainsworth"]#["beatles", "ballroom", "hainsworth"]#, "rwc_popular"]

# set the seed
pl.seed_everything(42)

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

# create the trainer
trainer = pl.Trainer.from_argparse_args(
    args,
    checkpoint_callback=checkpoint_callback,
    resume_from_checkpoint=checkpoint_path
)

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
                                    half=True if args.precision == 16 else False,
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
                                 half=True if args.precision == 16 else False,
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
                                                pin_memory=True)
val_dataloader = torch.utils.data.DataLoader(val_dataset_list, 
                                            shuffle=args.shuffle,
                                            batch_size=1,
                                            num_workers=args.num_workers,
                                            pin_memory=False)    

# create the model with args
dict_args = vars(args)
dict_args["nparams"] = 2
dict_args["target_sample_rate"] = args.audio_sample_rate / args.target_factor

model = model(**dict_args)

# summary 
torchsummary.summary(model, [(1,args.train_length)], device="cpu")

if __name__ == '__main__':
    # train!
    #for dl in train_dataloader:
    #    print(dl[0].shape, dl[1].shape)
    #raise ValueError
    trainer.fit(model, train_dataloader, val_dataloader)
