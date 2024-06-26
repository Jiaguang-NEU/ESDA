import os
import random
import time
import datetime
import cv2
import numpy as np
import math
import logging
import argparse
import os.path as osp
import glob
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
from PIL import Image



from model import ESDA
from util import dataset
from util import transform, config
from util.util import AverageMeter, poly_learning_rate, intersectionAndUnionGPU, setup_seed, \
                    check_makedirs, get_model_para_number,fix_bn

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)
val_manual_seed = 456
val_num = 1
setup_seed(val_manual_seed, False)
seed_array = np.random.randint(0, 1000, val_num)  # seed->[0,999]


def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch Semantic Segmentation')
    parser.add_argument('--viz', action='store_true', default=False)
    parser.add_argument('--config', type=str,
                        default='/ESDA-main/config/pascal/pascal_split0_vit_b_16_480.yaml',
                        help='config file')
    parser.add_argument('opts', help='see config/ade20k/ade20k_pspnet50.yaml for all options', default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--local_rank', type=int, default=-1,
                        help='number of cpu threads to use during batch generation')
    parser.add_argument('--exp_path', type=str,
                        default='/ESDA-main/exp/Test/pascal/ESDA')
    parser.add_argument('--snapshot_path', type=str,
                        default='/ESDA-main/exp/Test/pascal/ESDA/snapshot')
    parser.add_argument('--result_path', type=str,
                        default='/ESDA-main/exp/Test/pascal/ESDA/result')
    parser.add_argument('--show_path', type=str,
                        default='/ESDA-main/exp/Test/pascal/ESDA/show')
    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    cfg = config.merge_cfg_from_args(cfg, args)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    return cfg


def get_logger():
    logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    return logger


def get_model(args):
    model = ESDA(args)

    model = model.cuda()

    # Resume
    check_makedirs(args.snapshot_path)
    check_makedirs(args.result_path)
    check_makedirs(args.show_path)

    if args.weight:
        weight_path = args.weight
        if os.path.isfile(weight_path):
            logger.info("=> loading checkpoint '{}'".format(weight_path))
            checkpoint = torch.load(weight_path, map_location=torch.device('cpu'))
            args.start_epoch = checkpoint['epoch']
            new_param = checkpoint['state_dict']
            try:
                model.load_state_dict(new_param)
            except RuntimeError:  # 1GPU loads mGPU model
                for key in list(new_param.keys()):
                    new_param[key[7:]] = new_param.pop(key)
                model.load_state_dict(new_param)
            logger.info("=> loaded checkpoint '{}' (epoch {})".format(weight_path, checkpoint['epoch']))
        else:
            logger.info("=> no checkpoint found at '{}'".format(weight_path))

    # Get model para.
    total_number, learnable_number = get_model_para_number(model)
    print('Number of Parameters: %d' % (total_number))
    print('Number of Learnable Parameters: %d' % (learnable_number))

    time.sleep(5)
    return model

def main_process():
    return not args.distributed or (args.distributed and (args.local_rank == 0))

def main():
    global args, logger, writer
    args = get_parser()
    logger = get_logger()
    args.distributed = True if torch.cuda.device_count() > 1 else False
    if main_process():
        print(args)

    if args.manual_seed is not None:
        setup_seed(args.manual_seed, args.seed_deterministic)

    if main_process():
        logger.info("=> creating model ...")
    model = get_model(args)
    if main_process():
        logger.info(model)
    if main_process() and args.viz:
        writer = SummaryWriter(args.result_path)

    # ----------------------  DATASET  ----------------------
    value_scale = 255
    mean = [0.485, 0.456, 0.406]
    mean = [item * value_scale for item in mean]
    std = [0.229, 0.224, 0.225]
    std = [item * value_scale for item in std]
    # Val
    if args.evaluate:
        if args.resized_val:
            val_transform = transform.Compose([
                transform.Resize(size=args.val_size),
                transform.ToTensor(),
                transform.Normalize(mean=mean, std=std)])
        else:
            val_transform = transform.Compose([
                transform.test_Resize(size=args.val_size),
                transform.ToTensor(),
                transform.Normalize(mean=mean, std=std)])
        val_data = dataset.SemData(split=args.split, data_root=args.data_root,
                                   data_list=args.val_list, transform=val_transform, mode='val',
                                   use_coco=args.use_coco, use_split_coco=args.use_split_coco)
        val_sampler = None
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size_val, shuffle=False,
                                                 num_workers=args.workers, pin_memory=False, sampler=None)

        # ----------------------  VAL  ----------------------
        start_time = time.time()
        FBIoU_array = np.zeros(val_num)
        mIoU_array = np.zeros(val_num)
        for val_id in range(val_num):
            val_seed = seed_array[val_id]
            print('Val: [{}/{}] \t Seed: {}'.format(val_id + 1, val_num, val_seed))
            fb_iou, miou = validate(val_loader, model, val_seed)
            FBIoU_array[val_id], mIoU_array[val_id] = fb_iou, miou,

        total_time = time.time() - start_time
        t_m, t_s = divmod(total_time, 60)
        t_h, t_m = divmod(t_m, 60)
        total_time = '{:02d}h {:02d}m {:02d}s'.format(int(t_h), int(t_m), int(t_s))

        print('\nTotal running time: {}'.format(total_time))
        print('Seed0: {}'.format(val_manual_seed))
        print('Seed:  {}'.format(seed_array))
        print('mIoU:  {}'.format(np.round(mIoU_array, 4)))
        print('FBIoU: {}'.format(np.round(FBIoU_array, 4)))
        print('-' * 43)
        print('Best_Seed_m: {} \t Best_Seed_F: {} '.format(seed_array[mIoU_array.argmax()], seed_array[FBIoU_array.argmax()]))
        print(
            'Best_mIoU: {:.4f} \t Best_FBIoU: {:.4f} '.format(mIoU_array.max(), FBIoU_array.max()))
        print(
            'Mean_mIoU: {:.4f} \t Mean_FBIoU: {:.4f} '.format(mIoU_array.mean(), FBIoU_array.mean()))

def validate(val_loader, model, val_seed):
    logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
    batch_time = AverageMeter()
    model_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    if args.use_coco:
        split_gap = 20
    else:
        split_gap = 5

    if args.split != 999:
        if args.use_coco:
            test_num = 1000  # 20000
        else:
            test_num = 1000  # 5000
    else:
        test_num = len(val_loader)
    class_intersection_meter = [0]*split_gap
    class_union_meter = [0]*split_gap

    if args.manual_seed is not None and args.fix_random_seed_val:
        setup_seed(val_seed, args.seed_deterministic)

    criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_label)

    model.eval()
    end = time.time()
    val_start = end


    assert test_num % args.batch_size_val == 0
    iter_num = 0
    db_epoch = math.ceil(test_num / (len(val_loader) - args.batch_size_val))

    for e in range(db_epoch):
        for i, (input, target, ori_label, img_cls, subcls) in enumerate(val_loader):
            if iter_num * args.batch_size_val >= test_num:
                break
            iter_num += 1
            data_time.update(time.time() - end)

            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            ori_label = ori_label.cuda(non_blocking=True)

            start_time = time.time()
            output = model(x=input, x_m=target, img_cls=img_cls)
            model_time.update(time.time() - start_time)

            if args.ori_resize:
                longerside = max(ori_label.size(1), ori_label.size(2))
                backmask = torch.ones(ori_label.size(0), longerside, longerside).cuda()*255
                backmask[0, :ori_label.size(1), :ori_label.size(2)] = ori_label
                target = backmask.clone().long()

            output = F.interpolate(output, size=target.size()[1:], mode='bilinear', align_corners=True)
            loss = criterion(output, target)

            output = output.max(1)[1]

            intersection, union, new_target = intersectionAndUnionGPU(output, target, args.classes, args.ignore_label)
            intersection, union, target, new_target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy(), new_target.cpu().numpy()
            intersection_meter.update(intersection), union_meter.update(union), target_meter.update(new_target)

            subcls = subcls[0].cpu().numpy()[0]  # subcls represent the index of support-image's class in the current epoch  # subcls is a "list" type, subcls[0].cpu().numpy() is array
            # class_intersection_meter[subcls] += intersection[1]
            # class_union_meter[subcls] += union[1]
            class_intersection_meter[(subcls)%split_gap] += intersection[1]  # val split0 value of subcls: 0~4 ; split3  value: 15~19
            class_union_meter[(subcls)%split_gap] += union[1]

            accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
            loss_meter.update(loss.item(), input.size(0))
            batch_time.update(time.time() - end)
            end = time.time()
            if ((i + 1) % (test_num/100) == 0) and main_process():
                logger.info('Test: [{}/{}] '
                            'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                            'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                            'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f}) '
                            'Accuracy {accuracy:.4f}.'.format(iter_num* args.batch_size_val, test_num,
                                                              data_time=data_time,
                                                              batch_time=batch_time,
                                                              loss_meter=loss_meter,
                                                              accuracy=accuracy))
    val_time = time.time() - val_start
    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)  # [IoU_B, IoU_F]
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)

    mIoU = np.mean(iou_class)  # so this is the FB-IoU
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)


    class_iou_class = []
    class_miou = 0
    for i in range(len(class_intersection_meter)):
        class_iou = class_intersection_meter[i]/(class_union_meter[i]+ 1e-10)
        class_iou_class.append(class_iou)
        class_miou += class_iou
    class_miou = class_miou*1.0 / len(class_intersection_meter)

    if main_process():
        logger.info('meanIoU---Val result: mIoU {:.4f}.'.format(class_miou))
        for i in range(split_gap):
            logger.info('Class_{} Result: iou {:.4f}.'.format(i + 1, class_iou_class[i]))
        logger.info('FBIoU---Val result: FBIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU, mAcc, allAcc))
        for i in range(args.classes):
            logger.info('Class_{} Result: iou/accuracy {:.4f}/{:.4f}.'.format(i, iou_class[i], accuracy_class[i]))
        logger.info('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')

    print('total time: {:.4f}, avg inference time: {:.4f}, count: {}'.format(val_time, model_time.avg, test_num))
    return mIoU, class_miou


if __name__ == '__main__':
    main()
