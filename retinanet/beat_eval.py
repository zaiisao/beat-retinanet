#import madmom
import mir_eval
import numpy as np
import scipy.signal
import torch
import os
from retinanet.utils import calc_iou

def find_beats(t, p, 
                smoothing=127, 
                threshold=0.5, 
                distance=None, 
                sample_rate=44100, 
                beat_type="beat",
                filter_type="none",
                peak_type="simple"):
    # 15, 2

    # t is ground truth beats
    # p is predicted beats 
    # 0 - no beat
    # 1 - beat

    N = p.shape[-1]

    if filter_type == "savgol":
        # apply smoothing with savgol filter
        p = scipy.signal.savgol_filter(p, smoothing, 2)   
    elif filter_type == "cheby":
        sos = scipy.signal.cheby1(10,
                                  1, 
                                  45, 
                                  btype='lowpass', 
                                  fs=sample_rate,
                                  output='sos')
        p = scipy.signal.sosfilt(sos, p)

    # normalize the smoothed signal between 0.0 and 1.0
    p /= np.max(np.abs(p))                                     

    if peak_type == "simple":
        # by default, we assume that the min distance between beats is fs/4
        # this allows for at max, 4 BPS, which corresponds to 240 BPM 
        # for downbeats, we assume max of 1 downBPS
        if beat_type == "beat":
            if distance is None:
                distance = sample_rate / 4
        elif beat_type == "downbeat":
            if distance is None:
                distance = sample_rate / 2
        else:
            raise RuntimeError(f"Invalid beat_type: `{beat_type}`.")

        # apply simple peak picking
        est_beats, heights = scipy.signal.find_peaks(p, height=threshold, distance=distance)

    elif peak_type == "cwt":
        # use wavelets
        est_beats = scipy.signal.find_peaks_cwt(p, np.arange(1,50))

    # compute the locations of ground truth beats
    ref_beats = np.squeeze(np.argwhere(t==1).astype('float32'))
    est_beats = est_beats.astype('float32')

    # compute beat points (samples) to seconds
    ref_beats /= float(sample_rate)
    est_beats /= float(sample_rate)

    # store the smoothed ODF
    est_sm = p

    return ref_beats, est_beats, est_sm

def evaluate(pred, target, target_sample_rate, use_dbn=False):

    t_beats = target[0,:]
    t_downbeats = target[1,:]
    p_beats = pred[0,:]
    p_downbeats = pred[1,:]

    # ref_beats, est_beats, _ = find_beats(t_beats.numpy(), 
    #                                     p_beats.numpy(), 
    #                                     beat_type="beat",
    #                                     sample_rate=target_sample_rate)

    # ref_downbeats, est_downbeats, _ = find_beats(t_downbeats.numpy(), 
    #                                             p_downbeats.numpy(), 
    #                                             beat_type="downbeat",
    #                                             sample_rate=target_sample_rate)

    # if use_dbn:
    #     beat_dbn = madmom.features.beats.DBNBeatTrackingProcessor(
    #         min_bpm=55,
    #         max_bpm=215,
    #         transition_lambda=100,
    #         fps=target_sample_rate,
    #         online=False)

    #     downbeat_dbn = madmom.features.beats.DBNBeatTrackingProcessor(
    #         min_bpm=10,
    #         max_bpm=75,
    #         transition_lambda=100,
    #         fps=target_sample_rate,
    #         online=False)

    #     beat_pred = pred[0,:].clamp(1e-8, 1-1e-8).view(-1).numpy()
    #     downbeat_pred = pred[1,:].clamp(1e-8, 1-1e-8).view(-1).numpy()

    #     est_beats = beat_dbn.process_offline(beat_pred)
    #     est_downbeats = downbeat_dbn.process_offline(downbeat_pred)

    # evaluate beats - trim beats before 5 seconds.
    ref_beats = mir_eval.beat.trim_beats(ref_beats)
    est_beats = mir_eval.beat.trim_beats(est_beats)
    beat_scores = mir_eval.beat.evaluate(ref_beats, est_beats)

    # evaluate downbeats - trim beats before 5 seconds.
    ref_downbeats = mir_eval.beat.trim_beats(ref_downbeats)
    est_downbeats = mir_eval.beat.trim_beats(est_downbeats)
    downbeat_scores = mir_eval.beat.evaluate(ref_downbeats, est_downbeats)

    return beat_scores, downbeat_scores

def compute_overlap(a, b):
    """
    Parameters
    ----------
    a: (N, 4) ndarray of float
    b: (K, 4) ndarray of float
    Returns
    -------
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    #area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    area = b[:, 1] - b[:, 0]

    #iw = np.minimum(np.expand_dims(a[:, 2], axis=1), b[:, 2]) - np.maximum(np.expand_dims(a[:, 0], 1), b[:, 0])
    iw = np.minimum(np.expand_dims(a[:, 1], axis=1), b[:, 1]) - np.maximum(np.expand_dims(a[:, 0], 1), b[:, 0])
    #ih = np.minimum(np.expand_dims(a[:, 3], axis=1), b[:, 3]) - np.maximum(np.expand_dims(a[:, 1], 1), b[:, 1])

    iw = np.maximum(iw, 0)
    #ih = np.maximum(ih, 0)

    #ua = np.expand_dims((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), axis=1) + area - iw * ih
    ua = np.expand_dims(a[:, 1] - a[:, 0], axis=1) + area - iw

    ua = np.maximum(ua, np.finfo(float).eps)

    #intersection = iw * ih
    intersection = iw

    return intersection / ua


def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.
    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def get_results_from_model(audio, target, model, iou_threshold=0.5, score_threshold=0.05):
    #data = dataset[index]
    #scale = data['scale']
    #audio, target = data

    if torch.cuda.is_available():
        # move data to GPU
        audio = audio.to('cuda')
        target = target.to('cuda')

    nblocks = len(model.module.dstcn.blocks)

    target_length = -(audio.size(dim=2) // -2**nblocks) * 2**nblocks
    audio_pad = (0, target_length - audio.size(dim=2))
    audio = torch.nn.functional.pad(audio, audio_pad, "constant", 0)

    # run network
    # scores, labels, boxes = model(audio.permute(2, 0, 1).cuda().float().unsqueeze(dim=0))
    # predicted_scores, predicted_labels, predicted_boxes = model((audio, target))

    predicted_scores, predicted_labels, predicted_boxes, losses = model(
        (audio, target),
        iou_threshold=iou_threshold,
        score_threshold=score_threshold
    )

    predicted_scores = predicted_scores.cpu()
    predicted_labels = predicted_labels.cpu()
    predicted_boxes  = predicted_boxes.cpu()
    
    return predicted_scores, predicted_labels, predicted_boxes, losses

def get_detections(dataloader, model, num_classes, score_threshold=0.05, max_detections=10000000):
    """ Get the detections from the retinanet using the generator.
    The result is a list of lists such that the size is:
        all_detections[num_images][num_classes] = detections[num_detections, 4 + num_classes]
    # Arguments
        dataset         : The generator used to run images through the retinanet.
        model           : The model to run on the images.
        score_threshold : The score confidence threshold to use.
        max_detections  : The maximum number of detections to use per image.
        save_path       : The path to save the images with visualized detections to.
    # Returns
        A list of lists containing the detections for each image in the generator.
    """
    all_detections = [[None for i in range(num_classes)] for j in range(len(dataloader))]

    model.eval()
    
    with torch.no_grad():
        for index, data in enumerate(dataloader):
            audio, target, metadata = data

            # if we have metadata, it is only during evaluation where batch size is always 1
            metadata = metadata[0]
        
            predicted_scores, predicted_labels, predicted_boxes, losses = get_results_from_model(audio, target, model, score_threshold=score_threshold)
            # scale = data['scale']

            # # run network
            # if torch.cuda.is_available():
            #     scores, labels, boxes = retinanet(data['img'].permute(2, 0, 1).cuda().float().unsqueeze(dim=0))
            # else:
            #     scores, labels, boxes = retinanet(data['img'].permute(2, 0, 1).float().unsqueeze(dim=0))
            # scores = scores.cpu().numpy()
            # labels = labels.cpu().numpy()
            # boxes  = boxes.cpu().numpy()

            # # correct boxes for image scale
            # boxes /= scale

            # select indices which have a score above the threshold
            indices = np.where(predicted_scores > score_threshold)[0]
            if indices.shape[0] > 0:
                # select those scores
                predicted_scores = predicted_scores[indices]

                # find the order with which to sort the scores
                scores_sort = np.argsort(-predicted_scores)[:max_detections]

                # select detections
                image_boxes      = predicted_boxes[indices[scores_sort], :]
                image_scores     = predicted_scores[scores_sort]
                image_labels     = predicted_labels[indices[scores_sort]]
                image_detections = np.concatenate([image_boxes, np.expand_dims(image_scores, axis=1), np.expand_dims(image_labels, axis=1)], axis=1)

                # copy detections to all_detections
                for label in range(num_classes):
                    all_detections[index][label] = image_detections[image_detections[:, -1] == label, :-1]
            else:
                # copy detections to all_detections
                for label in range(num_classes):
                    #all_detections[index][label] = np.zeros((0, 5))
                    all_detections[index][label] = np.zeros((0, 3))

            print('{}/{}'.format(index + 1, len(dataloader)), end='\r')

    return all_detections


def get_annotations(dataloader, num_classes):
    """ Get the ground truth annotations from the generator.
    The result is a list of lists such that the size is:
        all_detections[num_images][num_classes] = annotations[num_detections, 5]
    # Arguments
        generator : The generator used to retrieve ground truth annotations.
    # Returns
        A list of lists containing the annotations for each image in the generator.
    """
    all_annotations = [[None for i in range(num_classes)] for j in range(len(dataloader))]

    #for i in range(len(dataloader)):
    for i, data in enumerate(dataloader):
        _, batch_annotations, _ = data
        annotations = batch_annotations[0].cpu().numpy()
        # load the annotations
        #annotations = dataloader.load_annotations(i)

        # copy detections to all_annotations
        for label in range(num_classes):
            #all_annotations[i][label] = annotations[annotations[:, 4] == label, :4].copy()
            all_annotations[i][label] = annotations[annotations[:, 2] == label, :2].copy()

        print('{}/{}'.format(i + 1, len(dataloader)), end='\r')

    return all_annotations

def evaluate_beat_ap(
    dataloader,
    model,
    num_classes=2,
    iou_threshold=0.5,
    score_threshold=0.05,
    max_detections=100
):
    all_detections = get_detections(dataloader, model, num_classes, score_threshold=score_threshold, max_detections=max_detections)
    all_annotations = get_annotations(dataloader, num_classes)
    
    average_precisions = {}

    for label in range(num_classes):
        false_positives = np.zeros((0,))
        true_positives  = np.zeros((0,))
        scores          = np.zeros((0,))
        num_annotations = 0.0

        for i in range(len(dataloader)):
            detections           = all_detections[i][label]
            annotations          = all_annotations[i][label]
            num_annotations     += annotations.shape[0]
            detected_annotations = []

            for d in detections:
                #scores = np.append(scores, d[4])
                scores = np.append(scores, d[2])

                if annotations.shape[0] == 0:
                    false_positives = np.append(false_positives, 1)
                    true_positives  = np.append(true_positives, 0)
                    continue

                overlaps            = compute_overlap(np.expand_dims(d, axis=0), annotations)
                assigned_annotation = np.argmax(overlaps, axis=1)
                max_overlap         = overlaps[0, assigned_annotation]

                if max_overlap >= iou_threshold and assigned_annotation not in detected_annotations:
                    false_positives = np.append(false_positives, 0)
                    true_positives  = np.append(true_positives, 1)
                    detected_annotations.append(assigned_annotation)
                else:
                    false_positives = np.append(false_positives, 1)
                    true_positives  = np.append(true_positives, 0)

        # no annotations -> AP for this class is 0 (is this correct?)
        if num_annotations == 0:
            average_precisions[label] = 0, 0
            continue

        # sort by score
        indices         = np.argsort(-scores)
        false_positives = false_positives[indices]
        true_positives  = true_positives[indices]

        # compute false positives and true positives
        false_positives = np.cumsum(false_positives)
        true_positives  = np.cumsum(true_positives)

        # compute recall and precision
        recall    = true_positives / num_annotations
        precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)

        # compute average precision
        average_precision  = compute_ap(recall, precision)
        average_precisions[label] = average_precision, num_annotations


    print('\nmAP:')
    for label in range(num_classes):
        #label_name = dataloader.label_to_name(label)
        if label == 0:
            label_name = "Downbeat"
        elif label == 1:
            label_name = "Beat"
        else:
            raise NotImplementedError

        print('{}: {}'.format(label_name, average_precisions[label][0]))
        print("Precision: ",precision[-1])
        print("Recall: ",recall[-1])

        # if save_path!=None:
        #     plt.plot(recall,precision)
        #     # naming the x axis 
        #     plt.xlabel('Recall') 
        #     # naming the y axis 
        #     plt.ylabel('Precision') 

        #     # giving a title to my graph 
        #     plt.title('Precision Recall curve') 

        #     # function to show the plot
        #     plt.savefig(save_path+'/'+label_name+'_precision_recall.jpg')

    return average_precisions

def evaluate_beat_f_measure(dataloader, model, audio_downsampling_factor, score_threshold=0.20):
    model.eval()
    
    with torch.no_grad():
        # start collecting results
        results = []
        #image_ids = []

        for index, data in enumerate(dataloader):
            audio, target, metadata = data

            # if we have metadata, it is only during evaluation where batch size is always 1
            metadata = metadata[0]
        
            predicted_scores, predicted_labels, predicted_boxes, losses = get_results_from_model(audio, target, model, score_threshold=score_threshold)

            #evaluate_ap(target, (predicted_scores, predicted_labels, predicted_boxes))

            # correct boxes for image scale
            #boxes /= scale

            # append image to list of processed images
            # image_ids.append(dataset.image_ids[index])

            # print progress
            #print('{}/{}'.format(index, len(dataset)), end='\r')

            #length = audio.size(dim=2) // 256
            #length = audio.size(dim=2) // 128

            # wavebeat_format_pred_left = torch.zeros((2, length)).to(audio.device)
            # wavebeat_format_pred_average = torch.zeros((2, length)).to(audio.device)
            # wavebeat_format_pred_right = torch.zeros((2, length)).to(audio.device)
            # wavebeat_format_pred_weighted = torch.zeros((2, length)).to(audio.device)

            # wavebeat_format_target = torch.zeros((2, length)).to(audio.device)

            # box_scores_left = torch.zeros((2, length)).to(audio.device)
            # box_scores_right = torch.zeros((2, length)).to(audio.device)

            beat_pred_left_positions = []
            downbeat_pred_left_positions = []

            beat_pred_right_positions = []
            downbeat_pred_right_positions = []

            first_pred_beat_index, first_pred_downbeat_index = None, None
            last_pred_beat_index, last_pred_downbeat_index = None, None
            last_target_beat_index, last_target_downbeat_index = None, None

            # construct pred tensor
            for box_id in range(predicted_boxes.shape[0]):
                predicted_score = float(predicted_scores[box_id]) # predicted_scores: (number of anchor positions,)
                predicted_label = int(predicted_labels[box_id]) # predicted_labels: (number of anchor positions,)
                predicted_box = predicted_boxes[box_id, :] # The shape of predicted_boxes is (number of anchor positions, 2)

                # scores are sorted, so we can break
                # but this filtering is redundant, as the filtering is done within the evaluation part of the model
                if predicted_score < score_threshold:
                    continue

                # if beat (label 1), first row (index 0)
                # if downbeat (label 0), second row (index 1)
                # row = 1 - label
                left_position_index = int(predicted_box[0])
                right_position_index = int(predicted_box[1])

                # We obtained the base level audio from the original audio sampled at 22050 Hz by downsampling audio_downsampling_factor
                # and on that base level audio the gt beat intervals are defined.
                #
                # left_position_index * audio_downsampling_factor is the position of the beat location on the base level audio.
                # In order to get the time of that location, we need to divide it by 22050 Hz

                if predicted_label == 0:
                    downbeat_pred_left_positions.append(left_position_index * audio_downsampling_factor / 22050)
                    downbeat_pred_right_positions.append(right_position_index * audio_downsampling_factor / 22050)
                elif predicted_label == 1:
                    beat_pred_left_positions.append(left_position_index * audio_downsampling_factor / 22050)
                    beat_pred_right_positions.append(right_position_index * audio_downsampling_factor / 22050)

                # wavebeat_format_pred_left[row, min(left_position_index, length - 1)] = 1
                # wavebeat_format_pred_right[row, min(right_position_index, length - 1)] = 1

                # box_scores_left[row, min(left_position_index, length - 1)] = score
                # box_scores_right[row, min(right_position_index, length - 1)] = score

                if predicted_label == 0 and (first_pred_downbeat_index is None or left_position_index < first_pred_downbeat_index):
                    first_pred_downbeat_index = left_position_index
                elif predicted_label == 1 and (first_pred_beat_index is None or left_position_index < first_pred_beat_index):
                    first_pred_beat_index = left_position_index

                if predicted_label == 0 and (last_pred_downbeat_index is None or right_position_index > last_pred_downbeat_index):
                    last_pred_downbeat_index = right_position_index
                elif predicted_label == 1 and (last_pred_beat_index is None or right_position_index > last_pred_beat_index):
                    last_pred_beat_index = right_position_index

            beat_scores = predicted_scores[predicted_labels == 1]
            beat_intervals = predicted_boxes[predicted_labels == 1][beat_scores >= score_threshold]
            if beat_scores.size(dim=0) == 0:
                sorted_beat_intervals = beat_intervals
            else:
                sorted_beat_intervals = beat_intervals[beat_intervals[:, 0].sort()[1]]
            beat_ious = torch.zeros(1, 0).to(sorted_beat_intervals.device)

            downbeat_scores = predicted_scores[predicted_labels == 0]
            downbeat_intervals = predicted_boxes[predicted_labels == 0][downbeat_scores >= score_threshold]
            if downbeat_scores.size(dim=0) == 0:
                sorted_downbeat_intervals = downbeat_intervals
            else:
                sorted_downbeat_intervals = downbeat_intervals[downbeat_intervals[:, 0].sort()[1]]
            downbeat_ious = torch.zeros(1, 0).to(downbeat_intervals.device)

            # # start mAP file generation
            # gt_beat_intervals = target[0, target[0, :, 2] == 1, :2]
            # gt_downbeat_intervals = target[0, target[0, :, 2] == 0, :2]

            # gt_interval_filename = metadata['Filename'].replace('/data/', '/gt_intervals/').replace('.wav', '.txt')
            # os.makedirs(os.path.dirname(gt_interval_filename), exist_ok=True)
            # gt_interval_file = open(gt_interval_filename, "w+")

            # for gt_beat_interval_index in range(gt_beat_intervals.size(dim=0)):
            #     gt_beat_interval = gt_beat_intervals[gt_beat_interval_index]
            #     gt_interval_file.write(f"beat {int(gt_beat_interval[0])} 0 {int(gt_beat_interval[1])} 1\n")

            # for gt_downbeat_interval_index in range(gt_downbeat_intervals.size(dim=0)):
            #     gt_downbeat_interval = gt_downbeat_intervals[gt_downbeat_interval_index]
            #     gt_interval_file.write(f"downbeat {int(gt_downbeat_interval[0])} 0 {int(gt_downbeat_interval[1])} 1\n")

            # gt_interval_file.close()
            
            # pred_interval_filename = metadata['Filename'].replace('/data/', '/pred_intervals/').replace('.wav', '.txt')
            # os.makedirs(os.path.dirname(pred_interval_filename), exist_ok=True)
            # pred_interval_file = open(pred_interval_filename, "w+")

            # for pred_beat_interval_index in range(sorted_beat_intervals.size(dim=0)):
            #     pred_beat_interval = sorted_beat_intervals[pred_beat_interval_index]
            #     pred_beat_score = float(predicted_scores[predicted_labels == 1][beat_intervals[:, 0].sort()[1]][pred_beat_interval_index])
            #     pred_interval_file.write(f"beat {pred_beat_score} {int(pred_beat_interval[0])} 0 {int(pred_beat_interval[1])} 1\n")

            # for pred_downbeat_interval_index in range(sorted_downbeat_intervals.size(dim=0)):
            #     pred_downbeat_interval = sorted_downbeat_intervals[pred_downbeat_interval_index]
            #     pred_downbeat_score = float(predicted_scores[predicted_labels == 0][downbeat_intervals[:, 0].sort()[1]][pred_downbeat_interval_index])
            #     pred_interval_file.write(f"downbeat {pred_downbeat_score} {int(pred_downbeat_interval[0])} 0 {int(pred_downbeat_interval[1])} 1\n")

            # pred_interval_file.close()
            # # end mAP file generation

            for beat_index, beat_interval in enumerate(sorted_beat_intervals[:-1]):
                next_beat_interval = sorted_beat_intervals[beat_index + 1]

                beat_iou = calc_iou(beat_interval[None], next_beat_interval[None])
                beat_ious = torch.cat((beat_ious, beat_iou), dim=1)

            for downbeat_index, downbeat_interval in enumerate(sorted_downbeat_intervals[:-1]):
                next_downbeat_interval = sorted_downbeat_intervals[downbeat_index + 1]

                downbeat_iou = calc_iou(downbeat_interval[None], next_downbeat_interval[None])
                downbeat_ious = torch.cat((downbeat_ious, downbeat_iou), dim=1)

            # print(f"Beat IoUs: {beat_ious}")
            # print(f"Downbeat IoUs: {downbeat_ious}")

            if last_pred_beat_index is not None:
                #wavebeat_format_pred_left[0, min(last_pred_beat_index, length - 1)] = 1
                beat_pred_left_positions.append(last_pred_beat_index * audio_downsampling_factor / 22050)

            if last_pred_downbeat_index is not None:
                #wavebeat_format_pred_left[1, min(last_pred_downbeat_index, length - 1)] = 1
                downbeat_pred_left_positions.append(last_pred_downbeat_index * audio_downsampling_factor / 22050)

            # if first_pred_beat_index is not None:
            #     wavebeat_format_pred_right[0, min(first_pred_beat_index, length - 1)] = 1

            # if first_pred_downbeat_index is not None:
            #     wavebeat_format_pred_right[1, min(first_pred_downbeat_index, length - 1)] = 1

            # box_scores_left = torch.nn.functional.pad(box_scores_left[box_scores_left != 0], (1, 1), "constant", 1)
            # box_scores_right = torch.nn.functional.pad(box_scores_right[box_scores_right != 0], (1, 1), "constant", 1)

            # positive_bbox_indices_left = wavebeat_format_pred_left.nonzero()
            # positive_bbox_indices_right = wavebeat_format_pred_right.nonzero()

            # left_weights = torch.clamp(positive_bbox_indices_left * box_scores_left[box_scores_left.nonzero()] / (
            #     positive_bbox_indices_left * box_scores_left[box_scores_left.nonzero()] +
            #     positive_bbox_indices_right * box_scores_right[box_scores_right.nonzero()]
            # ), min=0.0, max=0.5)

            # right_weights = torch.clamp(positive_bbox_indices_right * box_scores_right[box_scores_right.nonzero()] / (
            #     positive_bbox_indices_left * box_scores_left[box_scores_left.nonzero()] +
            #     positive_bbox_indices_right * box_scores_right[box_scores_right.nonzero()]
            # ), min=0.0, max=0.5)

            # average_indices = torch.round(positive_bbox_indices_left * 0.5 + positive_bbox_indices_right * 0.5)
            # for index_pair in average_indices:
            #     wavebeat_format_pred_average[index_pair[0].long(), index_pair[1].long()] = 1

            # weighted_indices = torch.round(positive_bbox_indices_left * left_weights + positive_bbox_indices_right * right_weights)
            # for index_pair in weighted_indices:
            #     index_pair[0] = 0 if torch.isnan(index_pair[0]) else 1
            #     index_pair[1] = torch.nan_to_num(index_pair[1], nan=0.5)
            #     wavebeat_format_pred_weighted[index_pair[0].long(), index_pair[1].long()] = 1


            beat_target_left_positions = []
            downbeat_target_left_positions = []
            # construct target tensor
            for beat_interval in target[0]:
                label = int(beat_interval[2])
                # row = 1 - label

                left_position_index = int(beat_interval[0])
                right_position_index = int(beat_interval[1])

                # wavebeat_format_target[row, min(left_position_index, length - 1)] = 1
                if label == 0:
                    downbeat_target_left_positions.append(left_position_index * audio_downsampling_factor / 22050)
                elif label == 1:
                    beat_target_left_positions.append(left_position_index * audio_downsampling_factor / 22050)

                if label == 0 and (last_target_downbeat_index is None or right_position_index > last_target_downbeat_index):
                    last_target_downbeat_index = right_position_index
                elif label == 1 and (last_target_beat_index is None or right_position_index > last_target_beat_index):
                    last_target_beat_index = right_position_index

            #wavebeat_format_target[0, min(last_target_beat_index, length - 1)] = 1
            #wavebeat_format_target[1, min(last_target_downbeat_index, length - 1)] = 1
            beat_target_left_positions.append(last_target_beat_index * audio_downsampling_factor / 22050)
            downbeat_target_left_positions.append(last_target_downbeat_index * audio_downsampling_factor / 22050)

            #target_sample_rate = 22050 // 256
            # target_sample_rate = 22050 // 128

            predicted_scores = predicted_scores.cpu()
            predicted_labels = predicted_labels.cpu()
            predicted_boxes  = predicted_boxes.cpu()
            
            # beat_scores_left, downbeat_scores_left = evaluate(wavebeat_format_pred_left.view(2,-1),  
            #                                         wavebeat_format_target.view(2,-1), 
            #                                         target_sample_rate,
            #                                         use_dbn=False)
            beat_target_left_positions = np.array(beat_target_left_positions)
            beat_pred_left_positions = np.array(beat_pred_left_positions)
            downbeat_target_left_positions = np.array(downbeat_target_left_positions)
            downbeat_pred_left_positions = np.array(downbeat_pred_left_positions)

            # beat_pred_right_positions = np.array(beat_pred_right_positions)
            # downbeat_pred_right_positions = np.array(downbeat_pred_right_positions)

            # beat_left_and_length = np.stack((
            #     beat_pred_left_positions,
            #     beat_pred_right_positions - beat_pred_left_positions
            # ), axis=1)
            # beat_left_and_length = beat_left_and_length[beat_left_and_length[:, 0].argsort()]
            #print(f"Beat time and length\n{beat_left_and_length}")

            beat_target_left_positions.sort()
            beat_pred_left_positions.sort()
            downbeat_target_left_positions.sort()
            downbeat_pred_left_positions.sort()

            # print(f"beat_pred_left_positions: {beat_pred_left_positions}")
            # print(f"beat_target_left_positions: {beat_target_left_positions}")
            # print(f"downbeat_pred_left_positions: {downbeat_pred_left_positions}")
            # print(f"downbeat_target_left_positions: {downbeat_target_left_positions}")

            beat_target_left_positions = mir_eval.beat.trim_beats(beat_target_left_positions)
            beat_pred_left_positions = mir_eval.beat.trim_beats(beat_pred_left_positions)
            beat_scores = mir_eval.beat.evaluate(beat_target_left_positions, beat_pred_left_positions)

            # evaluate downbeats - trim beats before 5 seconds.
            downbeat_target_left_positions = mir_eval.beat.trim_beats(downbeat_target_left_positions)
            downbeat_pred_left_positions = mir_eval.beat.trim_beats(downbeat_pred_left_positions)
            downbeat_scores = mir_eval.beat.evaluate(downbeat_target_left_positions, downbeat_pred_left_positions)

            # try:
            #     dbn_beat_scores_left, dbn_downbeat_scores_left = evaluate(wavebeat_format_pred_left.view(2,-1), 
            #                                             wavebeat_format_target.view(2,-1), 
            #                                             target_sample_rate,
            #                                             use_dbn=True)
            # except:
            #     dbn_beat_scores_left = { 'F-measure': 0 }
            #     dbn_downbeat_scores_left = { 'F-measure': 0 }

            # beat_scores_right, downbeat_scores_right = evaluate(wavebeat_format_pred_right.view(2,-1),  
            #                                         wavebeat_format_target.view(2,-1), 
            #                                         target_sample_rate,
            #                                         use_dbn=False)

            # try:
            #     dbn_beat_scores_right, dbn_downbeat_scores_right = evaluate(wavebeat_format_pred_right.view(2,-1), 
            #                                             wavebeat_format_target.view(2,-1), 
            #                                             target_sample_rate,
            #                                             use_dbn=True)
            # except:
            #     dbn_beat_scores_right = { 'F-measure': 0 }
            #     dbn_downbeat_scores_right = { 'F-measure': 0 }

            # beat_scores_average, downbeat_scores_average = evaluate(wavebeat_format_pred_average.view(2,-1),  
            #                                         wavebeat_format_target.view(2,-1), 
            #                                         target_sample_rate,
            #                                         use_dbn=False)

            # try:
            #     dbn_beat_scores_average, dbn_downbeat_scores_average = evaluate(wavebeat_format_pred_average.view(2,-1), 
            #                                             wavebeat_format_target.view(2,-1), 
            #                                             target_sample_rate,
            #                                             use_dbn=True)
            # except:
            #     dbn_beat_scores_average = { 'F-measure': 0 }
            #     dbn_downbeat_scores_average = { 'F-measure': 0 }

            # beat_scores_weighted, downbeat_scores_weighted = evaluate(wavebeat_format_pred_weighted.view(2,-1),  
            #                                         wavebeat_format_target.view(2,-1), 
            #                                         target_sample_rate,
            #                                         use_dbn=False)

            # try:
            #     dbn_beat_scores_weighted, dbn_downbeat_scores_weighted = evaluate(wavebeat_format_pred_weighted.view(2,-1), 
            #                                             wavebeat_format_target.view(2,-1), 
            #                                             target_sample_rate,
            #                                             use_dbn=True)
            # except:
            #     dbn_beat_scores_weighted = { 'F-measure': 0 }
            #     dbn_downbeat_scores_weighted = { 'F-measure': 0 }


            print(f"{index}/{len(dataloader)} {metadata['Filename']}")
            if len(losses) == 3:
                print(f"BEAT (F-measure): {beat_scores['F-measure']:0.3f} | DOWNBEAT (F-measure): {downbeat_scores['F-measure']:0.3f} | CLS: {losses[0]:0.3f} | REG: {losses[1]:0.3f} | ADJ: {losses[2]:0.3f}")
            elif len(losses) == 4:
                print(f"BEAT (F-measure): {beat_scores['F-measure']:0.3f} | DOWNBEAT (F-measure): {downbeat_scores['F-measure']:0.3f} | CLS: {losses[0]:0.3f} | REG: {losses[1]:0.3f} | LFT: {losses[2]:0.3f} | ADJ: {losses[3]:0.3f}")
            #print("LEFT")
            # print(f"BEAT (F-measure): {beat_scores_left['F-measure']:0.3f} | DOWNBEAT (F-measure): {downbeat_scores_left['F-measure']:0.3f}")
            # print(f"(DBN)  BEAT (F-measure): {dbn_beat_scores_left['F-measure']:0.3f} | DOWNBEAT (F-measure): {dbn_downbeat_scores_left['F-measure']:0.3f}")
            #print("RIGHT")
            #print(f"BEAT (F-measure): {beat_scores_right['F-measure']:0.3f} | DOWNBEAT (F-measure): {downbeat_scores_right['F-measure']:0.3f}")
            # print(f"(DBN)  BEAT (F-measure): {dbn_beat_scores_right['F-measure']:0.3f} | DOWNBEAT (F-measure): {dbn_downbeat_scores_right['F-measure']:0.3f}")
            # print("AVERAGE")
            # print(f"BEAT (F-measure): {beat_scores_average['F-measure']:0.3f} | DOWNBEAT (F-measure): {downbeat_scores_average['F-measure']:0.3f}")
            # print(f"(DBN)  BEAT (F-measure): {dbn_beat_scores_average['F-measure']:0.3f} | DOWNBEAT (F-measure): {dbn_downbeat_scores_average['F-measure']:0.3f}")
            # print("WEIGHTED")
            # print(f"BEAT (F-measure): {beat_scores_weighted['F-measure']:0.3f} | DOWNBEAT (F-measure): {downbeat_scores_weighted['F-measure']:0.3f}")
            # print(f"(DBN)  BEAT (F-measure): {dbn_beat_scores_weighted['F-measure']:0.3f} | DOWNBEAT (F-measure): {dbn_downbeat_scores_weighted['F-measure']:0.3f}\n")

            # beat_scores = beat_scores_left
            # downbeat_scores = downbeat_scores_left
            # dbn_beat_scores = dbn_beat_scores_left
            # dbn_downbeat_scores = dbn_downbeat_scores_left

            # if predicted_boxes.shape[0] > 0:
            #     # change to (x, y, w, h) (MS COCO standard)
            #     #boxes[:, 2] -= boxes[:, 0]
            #     #boxes[:, 3] -= boxes[:, 1]

            #     # compute predicted labels and scores
            #     #for box, score, label in zip(boxes[0], scores[0], labels[0]):
            #     for box_id in range(predicted_boxes.shape[0]):
            #         predicted_score = float(predicted_scores[box_id])
            #         predicted_label = int(predicted_labels[box_id])
            #         predicted_box = predicted_boxes[box_id, :]

            #         # scores are sorted, so we can break
            #         if predicted_score < score_threshold:
            #             # break
            #             continue

            # append detection for each positively labeled class
            image_result = {
                'image_id': metadata["Filename"],
                #'category_id': dataset.label_to_coco_label(label),
                'score': float(predicted_score),
                'bbox': predicted_box.tolist(),
                'beat_scores': beat_scores,
                'downbeat_scores': downbeat_scores,
                'cls_loss': losses[0],
                'reg_loss': losses[1],
                'lft_loss': None if len(losses) == 2 else losses[2],
                'adj_loss': losses[2] if len(losses) == 2 else losses[3],

                # 'beat_scores_left': beat_scores_left,
                # 'downbeat_scores_left': downbeat_scores_left,
                # 'beat_scores_right': beat_scores_right,
                # 'downbeat_scores_right': downbeat_scores_right,
                # 'beat_scores_average': beat_scores_average,
                # 'downbeat_scores_average': downbeat_scores_average,
                # 'beat_scores_weighted': beat_scores_weighted,
                # 'downbeat_scores_weighted': downbeat_scores_weighted,
                # 'dbn_beat_scores': dbn_beat_scores,
                # 'dbn_downbeat_scores': dbn_downbeat_scores
            }

            # append detection to results
            results.append(image_result)
        # END for index, data in enumerate(dataset)

        # if not len(results):
        #     return

        beat_mean_f_measure = np.mean([result['beat_scores']['F-measure'] for result in results])  #MJ: results = detection results
        downbeat_mean_f_measure = np.mean([result['downbeat_scores']['F-measure'] for result in results])
        cls_loss_mean = np.mean([result['cls_loss'] for result in results])
        reg_loss_mean = np.mean([result['reg_loss'] for result in results])
        lft_loss_mean = np.mean([result['lft_loss'] for result in results])
        adj_loss_mean = np.mean([result['adj_loss'] for result in results])
        # left_beat_mean_f_measure = np.mean([result['beat_scores_left']['F-measure'] for result in results])
        # left_downbeat_mean_f_measure = np.mean([result['downbeat_scores_left']['F-measure'] for result in results])
        #right_beat_mean_f_measure = np.mean([result['beat_scores_right']['F-measure'] for result in results])
        #right_downbeat_mean_f_measure = np.mean([result['downbeat_scores_right']['F-measure'] for result in results])
        # average_beat_mean_f_measure = np.mean([result['beat_scores_average']['F-measure'] for result in results])
        # average_downbeat_mean_f_measure = np.mean([result['downbeat_scores_average']['F-measure'] for result in results])
        # weighted_beat_mean_f_measure = np.mean([result['beat_scores_weighted']['F-measure'] for result in results])
        # weighted_downbeat_mean_f_measure = np.mean([result['downbeat_scores_weighted']['F-measure'] for result in results])
        #dbn_beat_mean_f_measure = np.mean([result['dbn_beat_scores']['F-measure'] for result in results])
        #dbn_downbeat_mean_f_measure = np.mean([result['dbn_downbeat_scores']['F-measure'] for result in results])

        print(f"Average beat F-measure: {beat_mean_f_measure:0.3f}")
        print(f"Average downbeat F-measure: {downbeat_mean_f_measure:0.3f}")
        print(f"Average losses | CLS: {cls_loss_mean:0.3f} | REG: {reg_loss_mean:0.3f} | LFT: {lft_loss_mean:0.3f} | ADJ: {adj_loss_mean:0.3f}")
        # print(f"Average left beat F-measure: {left_beat_mean_f_measure:0.3f}")
        # print(f"Average left downbeat F-measure: {left_downbeat_mean_f_measure:0.3f}")
        #print(f"Average right beat F-measure: {right_beat_mean_f_measure:0.3f}")
        #print(f"Average right downbeat F-measure: {right_downbeat_mean_f_measure:0.3f}")
        # print(f"Average average beat F-measure: {average_beat_mean_f_measure:0.3f}")
        # print(f"Average average downbeat F-measure: {average_downbeat_mean_f_measure:0.3f}")
        # print(f"Average weighted beat F-measure: {weighted_beat_mean_f_measure:0.3f}")
        # print(f"Average weighted downbeat F-measure: {weighted_downbeat_mean_f_measure:0.3f}")
        print()
        # print(f"(DBN) Average beat score: {dbn_beat_mean_f_measure:0.3f}")
        # print(f"(DBN) Average downbeat score: {dbn_downbeat_mean_f_measure:0.3f}")

        model.train()

        # beat_mean_f_measure = left_beat_mean_f_measure#average_beat_mean_f_measure
        # downbeat_mean_f_measure = left_downbeat_mean_f_measure#average_downbeat_mean_f_measure
        return beat_mean_f_measure, downbeat_mean_f_measure, results#dbn_beat_mean_f_measure, dbn_downbeat_mean_f_measure
