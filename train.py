import os
import argparse
import glob
import logging
import pprint
import shutil
import time
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from typing import List
from bisect import bisect_right
from torch.autograd import Variable

from torch import Tensor

from config.config import get_cfg
from dataset import build_data_loader

from CoSODNet import CoSODNet

import transforms as trans

import evaluation.metric as M


def get_args_parser():
    """
    Parse arguments
    """

    parser = argparse.ArgumentParser("CoSOD_Train", add_help=False)
    parser.add_argument("-config_file", default="./config/cosod.yaml", metavar="FILE",
                        help="path to config file")
    parser.add_argument("-num_works", default=1, type=int)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("-model_root_dir", default="./checkpoint",
                        help="dir for saving checkpoint")
    parser.add_argument("-batch_size", default=1, type=int)
    parser.add_argument("-device_id", type=str, default="1", help="choose cuda visiable devices")
    parser.add_argument("-img_root", type=str, default="./dataset/train_data/DUTS_class/img")
    parser.add_argument("-co_gt_root", type=str, default="./dataset/train_data/DUTS_class/gt")
    parser.add_argument("-img_root_coco", type=str, default="./dataset/train_data/CoCo9k/img")
    parser.add_argument("-co_gt_root_coco", type=str, default="./dataset/train_data/CoCo9k/gt")
    parser.add_argument("-img_syn_root", type=str, default="./dataset/train_data/DUTS_class_syn/"
                                                           "img_png_seamless_cloning_add_naive/img")
    parser.add_argument("-img_rev_syn_root", type=str, default="./dataset/train_data/DUTS_class_syn/"
                                                               "img_png_seamless_cloning_add_naive_reverse_2/img")
    parser.add_argument("-co_gt_rev_syn_root", type=str, default="./dataset/train_data/DUTS_class_syn/"
                                                                 "img_png_seamless_cloning_add_naive_reverse_2/gt")
    parser.add_argument("-test_data_root", type=str, default="./dataset/test_data")
    parser.add_argument("-test_datasets", nargs='+', default=["CoCA"])
    parser.add_argument("-save_dir", type=str, default='./prediction')
    parser.add_argument("-train_w_coco_prob", type=float, default=0.5)
    parser.add_argument("-max_num", type=int, default=8)
    parser.add_argument("-test_max_num", type=int, default=25)
    parser.add_argument("-img_size", type=int, default=256)
    parser.add_argument("-scale_size", type=int, default=288)
    parser.add_argument("-train_steps", type=int, default=80000)
    parser.add_argument("-lr", type=float, default=1e-4)
    parser.add_argument("-STEPS", nargs='+', default=[60000, 80000])
    parser.add_argument("-GAMMA", type=float, default=0.1)
    parser.add_argument("-warmup_factor", type=float, default=1.0/1000)
    parser.add_argument("-warmup_iters", type=int, default=1000)
    parser.add_argument("-warmup_method", type=str, default="linear")
    parser.add_argument("-max_epoches", type=int, default=300)
    return parser


def save_loss(save_dir, whole_iter_num, epoch_total_loss, epoch_loss, epoch):
    fh = open(save_dir, 'a')
    fh.write('until_{}_run_iter_num{}\n'.format(epoch, whole_iter_num))
    fh.write('{}_epoch_total_loss:{}\n'.format(epoch, epoch_total_loss))
    fh.write('{}_epoch_loss:{}\n'.format(epoch, epoch_loss))
    fh.write('\n')
    fh.close()


def adjust_learning_rate(optimizer, decay_rate=.1):
    update_lr_group = optimizer.param_groups
    for param_group in update_lr_group:
        print('before lr: {}'.format(param_group['lr']))
        param_group['lr'] = param_group['lr'] * decay_rate
        print('after lr: {}'.format(param_group['lr']))
    return optimizer


def save_lr(save_dir, optimizer):
    update_lr_group = optimizer.param_groups[0]
    fh = open(save_dir, 'a')
    fh.write('encode:update:lr{}\n'.format(update_lr_group['lr']))
    fh.write('decode:update:lr{}\n'.format(update_lr_group['lr']))
    fh.write('\n')
    fh.close()


def create_logger(model_name):
    time_str = time.strftime('%Y-%m-%d-%H-%M')
    if not os.path.exists('./log/{}'.format(model_name)):
        os.makedirs('./log/{}'.format(model_name), exist_ok=True)
    log_file = './log/{}/{}_.log'.format(model_name, time_str)
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(
        filename=str(log_file),
        format=head,
    )
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    return logger


def _get_cfg(cfg_file):
    cfg = get_cfg()
    cfg.merge_from_file(cfg_file)
    cfg.freeze()

    return cfg


def _get_project_save_dir(model_root_dir, model_name):
    proj_save_dir = os.path.join(model_root_dir, model_name)

    if not os.path.exists(proj_save_dir):
        os.makedirs(proj_save_dir, exist_ok=True)

    return proj_save_dir


def build_optimizer(args, model: torch.nn.Module) -> torch.optim.Optimizer:

    base_params = [params for name, params in model.named_parameters()
                   if 'encoder' in name and params.requires_grad]
    other_params = [params for name, params in model.named_parameters()
                    if 'encoder' not in name]

    optimizer = torch.optim.Adam(
        [{'params': base_params, 'lr': args.lr * 0.01}, {'params': other_params}],
        lr=args.lr,
        betas=(0.9, 0.99)
    )

    return optimizer


class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
            self,
            optimizer: torch.optim.Optimizer,
            milestones: List[int],
            gamma: float = 0.1,
            warmup_factor: float = 0.001,
            warmup_iters: int = 1000,
            warmup_method: str = "linear",
            last_epoch: int = -1,
    ):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of" " increasing integers. Got {}", milestones
            )
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        warmup_factor = _get_warmup_factor_at_iter(
            self.warmup_method, self.last_epoch, self.warmup_iters, self.warmup_factor
        )
        return [
            base_lr * warmup_factor * self.gamma ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]

    def _compute_values(self) -> List[float]:
        return self.get_lr()


def _get_warmup_factor_at_iter(
        method: str, iter: int, warmup_iters: int, warmup_factor: float
) -> float:
    """
    Return the learning rate warmup factor at a specific iteration.
    See https://arxiv.org/abs/1706.02677 for more details.

    Args:
         method (str): warmup method; either "constant" or "linear".
         iter (int): iteration at which to calculate the warmup factor.
         warmup_iters (int): the number of warmup iterations.
         warmup_factor (float): the base warmup factor (the meaning changes according
            to the method used)

    Returns:
        float: the effective warmup factor at the given iteration.
    """
    if iter >= warmup_iters:
        return 1.0

    if method == "constant":
        return warmup_factor
    elif method == "linear":
        alpha = iter / warmup_iters
        return warmup_factor * (1 - alpha) + alpha
    else:
        raise ValueError("Unknown warmup method: {}".format(method))


def build_lr_scheduler(
        args, optimizer: torch.optim.Optimizer
) -> torch.optim.lr_scheduler._LRScheduler:
    return WarmupMultiStepLR(
        optimizer,
        args.STEPS,
        args.GAMMA,
        warmup_factor=args.warmup_factor,
        warmup_iters=args.warmup_iters,
        warmup_method=args.warmup_method
    )


class Criterion(nn.Module):
    def __init__(self):
        super(Criterion, self).__init__()

        self.bce = nn.BCEWithLogitsLoss()

        self.s_co_bce = 0
        # self.s_noco_bce = 0
        self.s_bg_bce = 0
        self.s_com_bce = 0
        self.s_co_1_bce = 0
        self.s_co_2_bce = 0
        self.s_iou = 0
        self.s_iou_com = 0

        self.f_co_bce = 0
        # self.f_noco_bce = 0
        self.f_bg_bce = 0
        # self.f_com_bce = 0
        self.f_iou = 0
        self.f_iou_com = 0

    def reset_loss(self):
        self.s_co_bce = 0
        # self.s_noco_bce = 0
        self.s_bg_bce = 0
        self.s_com_bce = 0
        self.s_co_1_bce = 0
        self.s_co_2_bce = 0
        self.s_iou = 0
        self.s_iou_com = 0

        self.f_co_bce = 0
        # self.f_noco_bce = 0
        self.f_bg_bce = 0
        # self.f_com_bce = 0
        self.f_iou = 0
        self.f_iou_com = 0

    def iou(self, pred, gt):
        pred = F.sigmoid(pred)
        N, C, H, W = pred.shape
        min_tensor = torch.where(pred < gt, pred, gt)
        max_tensor = torch.where(pred > gt, pred, gt)
        min_sum = min_tensor.view(N, C, H * W).sum(dim=2)
        max_sum = max_tensor.view(N, C, H * W).sum(dim=2)
        loss = 1 - (min_sum / max_sum).mean()
        return loss

    def stage_loss(self, stage_co_pred, stage_bg_pred,
                   stage_com_pred, stage_co_pred_1, stage_co_pred_2, co_gt,
                   bg_gt):
        pred_size = stage_co_pred.shape[2:]
        co_gt = F.interpolate(co_gt, size=pred_size, mode="nearest")
        # noco_gt = F.interpolate(noco_gt, size=pred_size, mode="nearest")
        bg_gt = F.interpolate(bg_gt, size=pred_size, mode="nearest")

        self.s_co_bce += self.bce(stage_co_pred, co_gt)
        # self.s_noco_bce += self.bce(stage_noco_pred, noco_gt)
        self.s_bg_bce += self.bce(stage_bg_pred, bg_gt)
        self.s_com_bce += self.bce(stage_com_pred, co_gt)
        self.s_co_1_bce += self.bce(stage_co_pred_1, co_gt)
        self.s_co_2_bce += self.bce(stage_co_pred_2, co_gt)
        self.s_iou += self.iou(stage_co_pred, co_gt)
        self.s_iou_com += self.iou(stage_com_pred, co_gt)

    def average_loss(self, stage_num):
        self.s_co_bce = self.s_co_bce / stage_num
        # self.s_noco_bce = self.s_noco_bce / stage_num
        self.s_bg_bce = self.s_bg_bce / stage_num
        self.s_com_bce = self.s_com_bce / stage_num
        self.s_co_1_bce = self.s_co_1_bce / stage_num
        self.s_co_2_bce = self.s_co_2_bce / stage_num
        self.s_iou = self.s_iou / stage_num
        self.s_iou_com = self.s_iou_com / stage_num

    def __call__(self, result, co_gt: Tensor):
        self.reset_loss()

        co_gt[co_gt < 0.5] = 0.
        co_gt[co_gt >= 0.5] = 1.

        bg_gt = 1 - co_gt

        co_pred = result.pop('co_pred')
        # noco_pred = result.pop('noco_pred')
        bg_pred = result.pop('bg_pred')
        com_pred = result.pop('com_pred')

        self.f_co_bce = self.bce(co_pred, co_gt)
        # self.f_noco_bce = self.bce(noco_pred, noco_gt)
        self.f_bg_bce = self.bce(bg_pred, bg_gt)
        self.f_com_bce = self.bce(com_pred, co_gt)
        self.f_iou = self.iou(co_pred, co_gt)
        self.f_iou_com = self.iou(com_pred, co_gt)

        stage_co_preds = result.pop('stage_co_preds')
        # stage_noco_preds = result.pop('stage_noco_preds')
        stage_bg_preds = result.pop('stage_bg_preds')
        stage_com_preds = result.pop('stage_com_preds')
        stage_co_preds_1 = result.pop('stage_co_preds_1')
        stage_co_preds_2 = result.pop('stage_co_preds_2')

        stage_num = len(stage_co_preds)

        for i in range(stage_num):
            self.stage_loss(
                stage_co_preds[i], stage_bg_preds[i],
                stage_com_preds[i], stage_co_preds_1[i], stage_co_preds_2[i],
                co_gt, bg_gt
            )

        self.average_loss(stage_num)

        loss = self.f_co_bce + self.f_bg_bce + self.f_com_bce + self.f_iou + self.f_iou_com + \
               self.s_co_bce + self.s_bg_bce + self.s_com_bce + self.s_co_1_bce + \
               self.s_co_2_bce + self.s_iou + self.s_iou_com

        return loss


def test_group(model, group_data, save_root, max_num):
    img_num = group_data['imgs'].shape[1]
    groups = list(range(0, img_num + 1, max_num))
    if groups[-1] != img_num:
        groups.append(img_num)

    print(groups)

    for i in range(len(groups) - 1):
        if i == len(groups) - 2:
            end = groups[i + 1]
            start = max(0, end - max_num)
        else:
            start = groups[i]
            end = groups[i + 1]

        print(start, end)

        inputs = Variable(group_data['imgs'][:, start:end].squeeze(0).cuda())
        subpaths = group_data['subpaths'][start:end]
        ori_sizes = group_data['ori_sizes'][start:end]

        # img_name = '_'.join(subpaths[0][0][:-4].split('/')).replace(' ', '_')
        with torch.no_grad():

            result = model(inputs)

            co_preds = result.pop("co_pred")
            pred_prob = torch.sigmoid(co_preds)

            save_final_path = os.path.join(save_root, subpaths[0][0].split('/')[0])
            os.makedirs(save_final_path, exist_ok=True)

            for p_id in range(end - start):
                pre = pred_prob[p_id, :, :, :].data.cpu()

                subpath = subpaths[p_id][0]
                ori_size = (ori_sizes[p_id][1].item(),
                            ori_sizes[p_id][0].item())

                transform = trans.Compose([
                    trans.ToPILImage(),
                    trans.Scale(ori_size)
                ])
                outputImage = transform(pre)
                filename = subpath.split('/')[1]
                outputImage.save(os.path.join(save_final_path, filename))


def main(args):
    cfg = _get_cfg(args.config_file)
    model_name = args.model_name
    if model_name is None:
        model_name = os.path.abspath('').split('/')[-1]
    proj_save_dir = _get_project_save_dir(args.model_root_dir, model_name)

    logger = create_logger(model_name)
    logger.info(pprint.pformat(args))
    logger.info(cfg)

    train_loader = build_data_loader(args, mode='train')

    logger.info('''
    Starting training:
        Train steps: {}
        Batch size: {}
        Learning rate: {}
        Training size: {}
    '''.format(args.train_steps, args.batch_size, args.lr, len(train_loader.dataset)))

    logger.info("=> building model")
    model = CoSODNet(cfg)

    model.cuda()
    model.train()

    logger.info(model)

    optimizer = build_optimizer(args, model)
    lr_scheduler = build_lr_scheduler(args, optimizer)

    max_epoches = args.max_epoches
    train_steps = args.train_steps

    cri = Criterion()

    whole_iter_num = 0
    for epoch in range(max_epoches):

        logger.info("Starting epoch {}/{}.".format(epoch + 1, max_epoches))
        logger.info("epoch: {} ------ lr:{}".format(epoch, optimizer.param_groups[1]['lr']))

        for iteration, data_batch in enumerate(train_loader):
            imgs = Variable(data_batch["imgs"].squeeze(0).cuda())
            co_gts = Variable(data_batch["co_gts"].squeeze(0).cuda())

            result = model(imgs)

            loss = cri(result, co_gts)

            optimizer.zero_grad()

            with torch.autograd.detect_anomaly():
                loss.backward()

            optimizer.step()
            lr_scheduler.step()

            whole_iter_num += 1

            if whole_iter_num == train_steps:
                torch.save(
                    model.state_dict(),
                    os.path.join(proj_save_dir, 'iterations{}.pth'.format(train_steps))
                )
                break

            logger.info('Whole iter step:{0} - epoch progress:{1}/{2} - total_loss:{3:.4f} - f_co_bce:{4:.4f} '
                        '- f_bg_bce: {5:.4f} - f_com_bce: {6:.4f} - f_iou: {7:.4f} - f_iou_com: {8:.4f} '
                        '- s_co_bce:{9:.4f} - s_bg_bce: {10:.4f} - s_com_bce: {11:.4f} - s_co_1_bce: {12:.4f} '
                        '- s_co_2_bce: {13:.4f} - s_iou:{14:.4f} - s_iou_com:{15:.4f} '
                        ' batch_size: {16}'.format(whole_iter_num, epoch, max_epoches,
                                                   loss.item(), cri.f_co_bce, cri.f_bg_bce, cri.f_com_bce,
                                                   cri.f_iou, cri.f_iou_com, cri.s_co_bce, cri.s_bg_bce,
                                                   cri.s_com_bce, cri.s_co_1_bce, cri.s_co_2_bce,
                                                   cri.s_iou, cri.s_iou_com, co_gts.shape[0]))


        Sm_fun = M.Smeasure()

        test_loaders = build_data_loader(args, mode='test')
        data_loader = test_loaders['CoCA']

        save_root = os.path.join(args.save_dir, 'CoCA', '{}_iter{}'.format(model_name, whole_iter_num))
        print("evaluating on {}".format('CoCA'))
        for idx, group_data in enumerate(data_loader):
            print('{}/{}'.format(idx, len(data_loader)))

            max_num = args.test_max_num
            flag = True
            while flag:
                try:
                    test_group(model, group_data, save_root, max_num)
                    flag = False
                except:
                    print("set max_num as {}".format(max_num-2))
                    max_num = max_num - 1
                    continue

        # pred_data_dir = os.path.join(save_root, dataset)
        label_data_dir = os.path.join(args.test_data_root, 'CoCA', 'GroundTruth')
        classes = os.listdir(label_data_dir)
        for k in range(len(classes)):
            print('\r{}/{}'.format(k, len(classes)), end="", flush=True)
            class_name = classes[k]
            img_list = os.listdir(os.path.join(label_data_dir, class_name))
            for l in range(len(img_list)):
                img_name = img_list[l]
                # print("{}/{}".format(class_name, img_name))
                pred = cv2.imread(os.path.join(save_root, class_name, img_name), 0)
                gt = cv2.imread(os.path.join(label_data_dir, class_name, img_name[:-4]+'.png'), 0)
                Sm_fun.step(pred=pred/255, gt=gt/255)

        sm = Sm_fun.get_results()['sm']
        logger.info('\nEvaluating epoch {0} get SM {1:.4f}'.format(epoch, sm))

        if sm > 0.72:
            torch.save(
                model.state_dict(),
                os.path.join(proj_save_dir, 'iterations{}.pth'.format(whole_iter_num))
            )
        else:
            shutil.rmtree(save_root)

    torch.save(
        model.state_dict(),
        os.path.join(proj_save_dir, 'iterations{}.pth'.format(whole_iter_num))
    )

    logger.info('Epoch finished !!!')


if __name__ == '__main__':
    ap = argparse.ArgumentParser("CoSOD training script", parents=[get_args_parser()])
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id
    cudnn.benchmark = True
    main(args)
