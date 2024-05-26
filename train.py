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


from model import ESDA
from util import dataset
from util import transform, config
from util.util import AverageMeter, poly_learning_rate, intersectionAndUnionGPU, setup_seed, \
     check_makedirs, get_model_para_number, fix_bn

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)


def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch Semantic Segmentation')
    parser.add_argument('--viz', action='store_true', default=True)
    parser.add_argument('--config', type=str,
                        default='=/ESDA-main/config/pascal/pascal_split0_vit_b_16_480.yaml', help='config file')
    parser.add_argument('opts', help='see config/ade20k/ade20k_pspnet50.yaml for all options', default=None, nargs=argparse.REMAINDER)
    parser.add_argument('--local_rank', type=int, default=-1,
                        help='number of cpu threads to use during batch generation')
    parser.add_argument('--exp_path', type=str,
                        default='/ESDA-main/exp/Train/pascal/ESDA')
    parser.add_argument('--snapshot_path', type=str,
                        default='/ESDA-main/exp/Train/pascal/ESDA/snapshot')
    parser.add_argument('--result_path', type=str,
                        default='/ESDA-main/exp/Train/pascal/ESDA/result')
    parser.add_argument('--show_path', type=str,
                        default='/ESDA-main/exp/Train/pascal/ESDA/show')
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

    for name, param in model.named_parameters():
        if ('text_cls_proj' in name) or ('prompt' in name) or ('decoder' in name):
            param.requires_grad = True
            print(name)
        else:
            param.requires_grad = False

    params = [p for p in model.parameters() if p.requires_grad]

    if len(args.replace_indices) != 0:
        optimizer = torch.optim.SGD(
            params,
            lr=args.base_lr, momentum=args.momentum, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(
            params,
            lr=args.base_lr, momentum=args.momentum, weight_decay=args.weight_decay)
    if args.distributed:
        # Initialize Process Group
        # os.environ['MASTER_ADDR'] = 'localhost'
        # os.environ['MASTER_PORT'] = '6648'
        # dist.init_process_group(backend='nccl', init_method='env://', rank=0, world_size=1)
        dist.init_process_group(backend='nccl')
        print('args.local_rank: ', args.local_rank)
        torch.cuda.set_device(args.local_rank)
        device = torch.device('cuda', args.local_rank)
        model.to(device)
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank)
    else:
        model = model.cuda()

    # Resume
    # get_save_path(args)
    check_makedirs(args.snapshot_path)
    check_makedirs(args.result_path)
    check_makedirs(args.show_path)

    if args.resume_path:
        resume_path = args.resume_path
        if os.path.isfile(resume_path):
            if main_process():
                logger.info("=> loading checkpoint '{}'".format(resume_path))
            checkpoint = torch.load(resume_path, map_location=torch.device('cpu'))
            args.start_epoch = checkpoint['epoch']
            new_param = checkpoint['state_dict']
            try:
                model.load_state_dict(new_param)
            except RuntimeError:  # 1GPU loads mGPU model
                for key in list(new_param.keys()):
                    new_param[key[7:]] = new_param.pop(key)
                model.load_state_dict(new_param)
            optimizer.load_state_dict(checkpoint['optimizer'])
            if main_process():
                logger.info("=> loaded checkpoint '{}' (epoch {})".format(resume_path, checkpoint['epoch']))
        else:
            if main_process():
                logger.info("=> no checkpoint found at '{}'".format(resume_path))

    # Get model para.
    total_number, learnable_number = get_model_para_number(model)
    if main_process():
        print('Number of Parameters: %d' % (total_number))
        print('Number of Learnable Parameters: %d' % (learnable_number))

    time.sleep(5)
    return model, optimizer

def worker_init_fn(worker_id):
    random.seed(args.manual_seed + worker_id)


def main_process():
    return not args.distributed or (args.distributed and (args.local_rank == 0))
    # 优先级 not>and>or

def main():
    global args, logger, writer
    args = get_parser()
    logger = get_logger()
    args.distributed = True if torch.cuda.device_count() > 1 else False
    if main_process():
        print(args)

    if args.manual_seed is not None:
        setup_seed(args.manual_seed, args.seed_deterministic)

    assert args.classes > 1
    assert args.zoom_factor in [1, 2, 4, 8]

    if main_process():
        logger.info("=> creating model ...")
    model, optimizer = get_model(args)
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

    assert args.split in [0, 1, 2, 3, 999]
    train_transform = [
        transform.RandScale([args.scale_min, args.scale_max]),
        transform.RandRotate([args.rotate_min, args.rotate_max], padding=mean, ignore_label=args.padding_label),
        transform.RandomGaussianBlur(),
        transform.RandomHorizontalFlip(),
        transform.Crop([args.train_h, args.train_w], crop_type='rand', padding=mean, ignore_label=args.padding_label),
        transform.ToTensor(),
        transform.Normalize(mean=mean, std=std)]
    train_transform = transform.Compose(train_transform)
    train_data = dataset.SemData(split=args.split, data_root=args.data_root,
                                data_list=args.train_list, transform=train_transform, mode='train',
                                use_coco=args.use_coco, use_split_coco=args.use_split_coco)
    train_sampler = DistributedSampler(train_data) if args.distributed else None
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, num_workers=args.workers,
                                               pin_memory=True, sampler=train_sampler, drop_last=True,
                                               shuffle=False if args.distributed else True)
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
    # ----------------------  TRAINVAL  ----------------------
    global max_iou, fb_iou, best_epoch, keep_epoch, val_num
    max_iou = 0.
    fb_iou = 0.
    best_epoch = 0
    keep_epoch = 0
    val_num = 0

    start_time = time.time()

    for epoch in range(args.start_epoch, args.epochs):
        if keep_epoch == args.stop_interval:
            break
        if args.fix_random_seed_val:
            setup_seed(args.manual_seed + epoch, args.seed_deterministic)

        epoch_log = epoch + 1
        keep_epoch += 1
        if args.distributed:
            train_sampler.set_epoch(epoch)
        # ----------------------  TRAIN  ----------------------
        loss_train, mIoU_train, mAcc_train, allAcc_train = train(train_loader, val_loader, model, optimizer, epoch)
        if main_process() and args.viz:
            writer.add_scalar('loss_train', loss_train, epoch_log)
            writer.add_scalar('mIoU_train', mIoU_train, epoch_log)
            writer.add_scalar('mAcc_train', mAcc_train, epoch_log)
            writer.add_scalar('allAcc_train', allAcc_train, epoch_log)

        # save model for <resuming>
        if (epoch % args.save_freq == 0) and (epoch > 0) and main_process():

            filename = args.snapshot_path + '/epoch_lastest.pth'
            logger.info('Saving checkpoint to: ' + filename)
            if osp.exists(filename):
                os.remove(filename)
            torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()},
                       filename)

        # -----------------------  VAL  -----------------------
        if args.evaluate and epoch % 1 == 0:
            loss_val, mIoU_val, mAcc_val, allAcc_val, class_miou = validate(val_loader, model)
            val_num += 1
            if main_process() and args.viz:
                writer.add_scalar('loss_val', loss_val, epoch_log)
                writer.add_scalar('mIoU_val', mIoU_val, epoch_log)
                writer.add_scalar('mAcc_val', mAcc_val, epoch_log)
                writer.add_scalar('class_miou_val', class_miou, epoch_log)
                writer.add_scalar('allAcc_val', allAcc_val, epoch_log)

            if class_miou > max_iou:
                max_iou = class_miou
                fb_iou = mIoU_val
                best_epoch = epoch
                keep_epoch = 0

                filename = args.snapshot_path + '/train_epoch_' + str(epoch) + '_miou_{:.4f}'.format(max_iou) + '_fb_iou_{:.4f}'.format(fb_iou) +'.pth'

                if main_process():
                    logger.info('Saving checkpoint to: ' + filename)
                    torch.save(
                        {'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()},
                        filename)

                file_list = glob.glob(args.snapshot_path + '/*')
                print(file_list)
                if file_list:
                    for f in file_list:
                        if f != filename and f != (args.snapshot_path + '/epoch_lastest.pth'):
                            os.remove(f)
                            print("Deleted " + str(f))

    total_time = time.time() - start_time
    t_m, t_s = divmod(total_time, 60)
    t_h, t_m = divmod(t_m, 60)
    total_time = '{:02d}h {:02d}m {:02d}s'.format(int(t_h), int(t_m), int(t_s))

    if main_process():
        print('\nEpoch: {}/{} \t Total running time: {}'.format(epoch_log, args.epochs, total_time))
        print('The number of models validated: {}'.format(val_num))
        print('\n<<<<<<<<<<<<<<<<<<<<<<<<<<<<<  Final Best Result   <<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
        print('\t Group:{} \t Best_step:{}'.format(args.split, best_epoch))
        print('mIoU:{:.4f} '.format(max_iou))
        print('>' * 80)
        print('%s' % datetime.datetime.now())


def train(train_loader, val_loader, model, optimizer, epoch):
    global max_iou, best_epoch, keep_epoch, val_num
    batch_time = AverageMeter()
    data_time = AverageMeter()
    main_loss_meter = AverageMeter()
    aux_loss1_meter = AverageMeter()
    aux_loss2_meter = AverageMeter()
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    model.train()
    if args.fix_bn:
        model.apply(fix_bn)  # fix batchnorm
    end = time.time()
    val_time = 0.
    max_iter = args.epochs * len(train_loader)
    if main_process():
        print('Warmup: {}'.format(args.warmup))

    for i, (input, target, img_cls, subcls) in enumerate(train_loader):
        data_time.update(time.time() - end)
        current_iter = epoch * len(train_loader) + i + 1
        poly_learning_rate(optimizer, args.base_lr, current_iter, max_iter, power=args.power,
                           index_split=args.index_split, warmup=args.warmup, warmup_step=len(train_loader) // 2)

        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        output, main_loss, aux_loss1, aux_loss2 = model(x=input, x_m=target, img_cls=img_cls)
        loss = args.loss_weight[0] * main_loss + args.loss_weight[1] * aux_loss1 + args.loss_weight[2] * aux_loss2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        n = input.size(0)  # batch_size
        intersection, union, target = intersectionAndUnionGPU(output, target, args.classes,args.ignore_label)

        intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
        intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)

        accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)  # allAcc
        main_loss_meter.update(main_loss.item(), n)
        aux_loss1_meter.update(aux_loss1.item(), n)
        aux_loss2_meter.update(aux_loss2.item(), n)
        loss_meter.update(loss.item(), n)
        batch_time.update(time.time() - end - val_time)
        end = time.time()

        remain_iter = max_iter - current_iter
        remain_time = remain_iter * batch_time.avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))

        if (i + 1) % args.print_freq == 0 and main_process():
            logger.info('Epoch: [{}/{}][{}/{}] '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        'Remain {remain_time} '
                        'MainLoss {main_loss_meter.val:.4f} '
                        'AuxLoss1 {aux_loss1_meter.val:.4f} '
                        'AuxLoss2 {aux_loss2_meter.val:.4f} ' 
                        'Loss {loss_meter.val:.4f} '
                        'Accuracy {accuracy:.4f}.'.format(epoch+1, args.epochs, i + 1, len(train_loader),
                                                          batch_time=batch_time,
                                                          data_time=data_time,
                                                          remain_time=remain_time,
                                                          main_loss_meter=main_loss_meter,
                                                          aux_loss1_meter=aux_loss1_meter,
                                                          aux_loss2_meter=aux_loss2_meter,
                                                          loss_meter=loss_meter,
                                                          accuracy=accuracy))
        if main_process() and args.viz:
            writer.add_scalar('loss_train_batch', main_loss_meter.val, current_iter)
            writer.add_scalar('mIoU_train_batch', np.mean(intersection / (union + 1e-10)), current_iter)
            writer.add_scalar('mAcc_train_batch', np.mean(intersection / (target + 1e-10)), current_iter)
            writer.add_scalar('allAcc_train_batch', accuracy, current_iter)

        # -----------------------  SubEpoch VAL  -----------------------
        if args.evaluate and args.SubEpoch_val and (args.epochs <= 100 and epoch % 1 == 0 and epoch > 0) and (
                i == round(len(train_loader) / 2)):  # <if> max_epoch<=100 <do> half_epoch Val
            loss_val, mIoU_val, mAcc_val, allAcc_val, class_miou = validate(val_loader, model)
            val_num += 1
            # save model for <testing>
            if class_miou > max_iou:
                max_iou = class_miou
                fb_iou = mIoU_val
                best_epoch = epoch
                keep_epoch = 0

                filename = args.snapshot_path + '/train_epoch_' + str(epoch) + '_miou_{:.4f}'.format(max_iou) + '_fb_iou_{:.4f}'.format(fb_iou) + '.pth'

                if main_process():
                    logger.info('Saving checkpoint to: ' + filename)
                    torch.save(
                        {'epoch': epoch - 0.5, 'state_dict': model.state_dict(),
                         'optimizer': optimizer.state_dict()},
                        filename)

                file_list = glob.glob(args.snapshot_path + '/*')
                print(file_list)
                if file_list:
                    for f in file_list:
                        if f != filename and f != (args.snapshot_path + '/epoch_lastest.pth'):
                            os.remove(f)
                            print("Deleted " + str(f))

            model.train()
            if args.fix_bn:
                model.apply(fix_bn)  # fix batchnorm

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)

    if main_process():
        logger.info('Train result at epoch [{}/{}]: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(epoch, args.epochs, mIoU, mAcc, allAcc))
        for i in range(args.classes):
            logger.info('Class_{} Result: iou/accuracy {:.4f}/{:.4f}.'.format(i, iou_class[i], accuracy_class[i]))
    return loss_meter.avg, mIoU, mAcc, allAcc


def validate(val_loader, model):
    if main_process():
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
        setup_seed(args.manual_seed, args.seed_deterministic)

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
    return loss_meter.avg, mIoU, mAcc, allAcc, class_miou


if __name__ == '__main__':
    main()
