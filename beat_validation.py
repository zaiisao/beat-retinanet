import argparse
import torch
import os
import glob
import tqdm
from torchvision import transforms

from retinanet import model
from retinanet.dataloader import BeatDataset
from retinanet import csv_eval

assert torch.__version__.split('.')[0] == '1'

print('CUDA available: {}'.format(torch.cuda.is_available()))

datasets = ["ballroom", "hainsworth", "carnatic"]
results = {}

def main(args=None):
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')
    parser.add_argument('--checkpoints_dir', type=str, default='./', help='Path to pre-trained model log directory with checkpoint.')
    parser.add_argument('--preload', action="store_true")
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--beatles_audio_dir', type=str, default='./data')
    parser.add_argument('--beatles_annot_dir', type=str, default='./data')
    parser.add_argument('--ballroom_audio_dir', type=str, default='./data')
    parser.add_argument('--ballroom_annot_dir', type=str, default='./data')
    parser.add_argument('--hainsworth_audio_dir', type=str, default='./data')
    parser.add_argument('--hainsworth_annot_dir', type=str, default='./data')
    parser.add_argument('--rwc_popular_audio_dir', type=str, default='./data')
    parser.add_argument('--rwc_popular_annot_dir', type=str, default='./data')
    parser.add_argument('--gtzan_audio_dir', type=str, default='./data')
    parser.add_argument('--gtzan_annot_dir', type=str, default='./data')
    parser.add_argument('--smc_audio_dir', type=str, default='./data')
    parser.add_argument('--smc_annot_dir', type=str, default='./data')
    parser.add_argument('--audio_sample_rate', type=int, default=22050)
    parser.add_argument('--channel_growth', type=int, default=32)
    parser.add_argument('--channel_width', type=int, default=32)
    parser.add_argument('--norm_type', type=str, default='BatchNorm')
    parser.add_argument('--act_type', type=str, default='PReLU')
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--length', type=int, default=2097152)
    parser.add_argument('--ninputs', type=int, default=1)
    parser.add_argument('--noutputs', type=int, default=2)
    parser.add_argument('--nblocks', type=int, default=10)
    parser.add_argument('--kernel_size', type=int, default=15)
    parser.add_argument('--stride', type=int, default=2)
    parser.add_argument('--target_factor', type=int, default=256)

    args = parser.parse_args()

    # find the checkpoint path
    ckpts = glob.glob(os.path.join(args.checkpoints_dir, "*.pt"))
    if len(ckpts) < 1:
        raise RuntimeError(f"No checkpoints found in {args.checkpoints_dir}.")
    else:
        ckpt_path = ckpts[-1]

    # Create the model
    #retinanet = model.resnet50(num_classes=dataset_val.num_classes(), pretrained=True)
    dict_args = vars(args)
    retinanet = model.resnet50(**dict_args)

    use_gpu = True

    if use_gpu:
        if torch.cuda.is_available():
            retinanet = retinanet.cuda()

    if torch.cuda.is_available():
        retinanet = torch.nn.DataParallel(retinanet).cuda()
    else:
        retinanet = torch.nn.DataParallel(retinanet)

    retinanet.load_state_dict(torch.load(
        ckpt_path,
        torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ))

    #dataset_val = CocoDataset(parser.coco_path, set_name='val2017',transform=transforms.Compose([Normalizer(), Resizer()]))
    # evaluate on each dataset using the test set
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
        elif dataset == "gtzan":
            audio_dir = args.gtzan_audio_dir
            annot_dir = args.gtzan_annot_dir
        elif dataset == "smc":
            audio_dir = args.smc_audio_dir
            annot_dir = args.smc_annot_dir

        test_dataset = BeatDataset(audio_dir,
                                        annot_dir,
                                        dataset=dataset,
                                        audio_sample_rate=args.audio_sample_rate,
                                        target_factor=args.target_factor,
                                        subset="test" if not dataset in ["gtzan", "smc"] else "full-val",
                                        augment=False,
                                        preload=args.preload,
                                        length=args.length)

        test_dataloader = torch.utils.data.DataLoader(test_dataset, 
                                                        shuffle=False,
                                                        batch_size=1,
                                                        num_workers=args.num_workers,
                                                        pin_memory=True)

        # setup tracking of metrics
        results[dataset] = {
            "F-measure" : {
                "beat" : [],
                "dbn beat" : [],
                "downbeat" : [],
                "dbn downbeat" : [],
            },
            "CMLt" : {
                "beat" : [],
                "dbn beat" : [],
                "downbeat" : [],
                "dbn downbeat" : [],
            },
            "AMLt" : {
                "beat" : [],
                "dbn beat" : [],
                "downbeat" : [],
                "dbn downbeat" : [],
            }
        }

        for example in tqdm(test_dataloader, ncols=80):
            audio, target, metadata = example

            # move data to GPU
            audio = audio.to('cuda:0')
            target = target.to('cuda:0')

            with torch.no_grad():
                pred = torch.sigmoid(model(audio))

            # move data back to CPU
            pred = pred.cpu()
            target = target.cpu()

            beat_scores, downbeat_scores = evaluate(pred.view(2,-1),  
                                                    target.view(2,-1), 
                                                    model.hparams.target_sample_rate,
                                                    use_dbn=False)

            dbn_beat_scores, dbn_downbeat_scores = evaluate(pred.view(2,-1), 
                                                    target.view(2,-1), 
                                                    model.hparams.target_sample_rate,
                                                    use_dbn=True)

            print()
            print(f"beat {beat_scores['F-measure']:0.3f} mean: {np.mean(results[dataset]['F-measure']['beat']):0.3f}  ")
            print(f"downbeat: {downbeat_scores['F-measure']:0.3f} mean: {np.mean(results[dataset]['F-measure']['downbeat']):0.3f}")

            results[dataset]['F-measure']['beat'].append(beat_scores['F-measure'])
            results[dataset]['CMLt']['beat'].append(beat_scores['Correct Metric Level Total'])
            results[dataset]['AMLt']['beat'].append(beat_scores['Any Metric Level Total'])

            results[dataset]['F-measure']['dbn beat'].append(dbn_beat_scores['F-measure'])
            results[dataset]['CMLt']['dbn beat'].append(dbn_beat_scores['Correct Metric Level Total'])
            results[dataset]['AMLt']['dbn beat'].append(dbn_beat_scores['Any Metric Level Total'])

            results[dataset]['F-measure']['downbeat'].append(downbeat_scores['F-measure'])
            results[dataset]['CMLt']['downbeat'].append(downbeat_scores['Correct Metric Level Total'])
            results[dataset]['AMLt']['downbeat'].append(downbeat_scores['Any Metric Level Total'])

            results[dataset]['F-measure']['dbn downbeat'].append(dbn_downbeat_scores['F-measure'])
            results[dataset]['CMLt']['dbn downbeat'].append(dbn_downbeat_scores['Correct Metric Level Total'])
            results[dataset]['AMLt']['dbn downbeat'].append(dbn_downbeat_scores['Any Metric Level Total'])

        print()
        print(f"{dataset}")
        print(f"F1 beat: {np.mean(results[dataset]['F-measure']['beat'])}   F1 downbeat: {np.mean(results[dataset]['F-measure']['downbeat'])}")
        print(f"CMLt beat: {np.mean(results[dataset]['CMLt']['beat'])}   CMLt downbeat: {np.mean(results[dataset]['CMLt']['downbeat'])}")
        print(f"AMLt beat: {np.mean(results[dataset]['AMLt']['beat'])}   AMLt downbeat: {np.mean(results[dataset]['AMLt']['downbeat'])}")
        print()
        print(f"F1 dbn beat: {np.mean(results[dataset]['F-measure']['dbn beat'])}   F1 dbn downbeat: {np.mean(results[dataset]['F-measure']['dbn downbeat'])}")
        print(f"CMLt dbn  beat: {np.mean(results[dataset]['CMLt']['dbn beat'])}   CMLt dbn downbeat: {np.mean(results[dataset]['CMLt']['dbn downbeat'])}")
        print(f"AMLt dbn beat: {np.mean(results[dataset]['AMLt']['dbn beat'])}   AMLt dbn downbeat: {np.mean(results[dataset]['AMLt']['dbn downbeat'])}")
        print()

    use_gpu = True

    if use_gpu:
        if torch.cuda.is_available():
            retinanet = retinanet.cuda()

    if torch.cuda.is_available():
        #retinanet.load_state_dict(torch.load(parser.model_path))
        retinanet = torch.nn.DataParallel(retinanet).cuda()
    else:
        retinanet.load_state_dict(torch.load(parser.model_path))
        retinanet = torch.nn.DataParallel(retinanet)

    retinanet.training = False
    retinanet.eval()
    retinanet.module.freeze_bn()

    print(csv_eval.evaluate(dataset_val, retinanet,iou_threshold=float(parser.iou_threshold)))



if __name__ == '__main__':
    main()
