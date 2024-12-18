#============================================
__author__ = "Sachin Mehta"
__maintainer__ = "Sachin Mehta"
#============================================
# import torch_xla.core.xla_model as xm
import time
import torch
from utilities.utils import AverageMeter
from utilities.metrics.classification_accuracy import accuracy
from torch.nn import functional as F
from utilities.print_utils import *

'''
Training loop
'''
def train(data_loader, model, criteria, optimizer, epoch, device='cuda'):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(data_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        input = input.to(device)
        target = target.to(device)

        # compute output
        output = model(input)

        # compute loss
        loss = criteria(output, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        #losses.update(loss.data[0], input.size(0))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1[0].item(), input.size(0))
        top5.update(prec5[0].item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 10 == 0: #print after every 100 batches
            print_log_message("Epoch: %d[%d/%d]\t\tBatch Time:%.4f\t\tLoss:%.4f\t\ttop1:%.4f (%.4f)\t\ttop5:%.4f (%.4f)" %
                              (epoch, i, len(data_loader), batch_time.avg, losses.avg, top1.val, top1.avg, top5.val, top5.avg))


    return top1.avg, losses.avg

'''
Validation loop
'''
def validate(data_loader, model, criteria=None, device='cuda'):
    batch_time = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # Access class labels from the dataset
    class_labels = getattr(data_loader.dataset, 'classes', None)
    num_classes = len(class_labels) if class_labels else len(set(data_loader.dataset.targets))

    per_class_correct = torch.zeros(num_classes, dtype=torch.int64, device=device)
    per_class_total = torch.zeros(num_classes, dtype=torch.int64, device=device)

    if criteria:
        losses = AverageMeter()

    # Switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(data_loader):
            input = input.to(device)
            target = target.to(device)

            # Compute output
            output = model(input)
            if criteria:
                loss = criteria(output, target)

            # Measure accuracy and record loss
            predictions = output.argmax(dim=1)
            for label, pred in zip(target, predictions):
                per_class_total[label] += 1
                if label == pred:
                    per_class_correct[label] += 1

            # Top-k accuracy
            (prec1, prec5), _ = accuracy(output, target, topk=(1, 5))
            top1.update(prec1[0].item(), input.size(0))
            top5.update(prec5[0].item(), input.size(0))

            # Measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 10 == 0:
                print_log_message(
                    "Batch:[%d/%d]\tBatchTime:%.3f\tTop1:%.3f (%.3f)\tTop5:%.3f(%.3f)" %
                    (i, len(data_loader), batch_time.avg, top1.val, top1.avg, top5.val, top5.avg))

        # Compute per-class accuracy
        per_class_acc = {
            cls: (100.0 * per_class_correct[cls].item() / per_class_total[cls].item() if per_class_total[cls] > 0 else 0.0)
            for cls in range(num_classes)
        }

        # Print per-class accuracies
        print("\nPer-Class Accuracy:")
        for cls, acc in per_class_acc.items():
            label = class_labels[cls] if class_labels else f"Class {cls}"
            print_info_message(f"{label}: {acc:.2f}%")

        print_info_message(' * Top1:%.3f Top5:%.3f' % (top1.avg, top5.avg))

        if criteria:
            return top1.avg, losses.avg, per_class_acc
        else:
            return top1.avg, per_class_acc



def train_multi(data_loader, model, criteria, optimizer, epoch, device='cuda'):
    batch_time = AverageMeter()
    losses = AverageMeter()
    prec = AverageMeter()
    rec = AverageMeter()

    # switch to train mode
    model.train()
    end = time.time()


    tp, fp, fn, tn, count = 0, 0, 0, 0, 0
    p_o, r_o, f_o = 0.0, 0.0, 0.0
    for i, (input, target) in enumerate(data_loader):

        target = target.to(device=device)
        target = target.max(dim=1)[0]
        # compute output
        output = model(input)
        loss = criteria(output, target.float()) * 80.0

        # measure accuracy and record loss
        pred = output.gt(0.0).long()

        tp += (pred + target).eq(2).sum(dim=0)
        fp += (pred - target).eq(1).sum(dim=0)
        fn += (pred - target).eq(-1).sum(dim=0)
        tn += (pred + target).eq(0).sum(dim=0)
        count += input.size(0)

        this_tp = (pred + target).eq(2).sum()
        this_fp = (pred - target).eq(1).sum()
        this_fn = (pred - target).eq(-1).sum()
        this_tn = (pred + target).eq(0).sum()
        this_acc = (this_tp + this_tn).float() / (this_tp + this_tn + this_fp + this_fn).float()

        this_prec = this_tp.float() / (this_tp + this_fp).float() * 100.0 if this_tp + this_fp != 0 else 0.0
        this_rec = this_tp.float() / (this_tp + this_fn).float() * 100.0 if this_tp + this_fn != 0 else 0.0

        losses.update(float(loss), input.size(0))
        prec.update(float(this_prec), input.size(0))
        rec.update(float(this_rec), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        # optimizer.step()
        xm.optimizer_step(optimizer, barrier=True)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        p_c = [float(tp[i].float() / (tp[i] + fp[i]).float()) * 100.0 if tp[i] > 0 else 0.0 for i in range(len(tp))]
        r_c = [float(tp[i].float() / (tp[i] + fn[i]).float()) * 100.0 if tp[i] > 0 else 0.0 for i in range(len(tp))]
        f_c = [2 * p_c[i] * r_c[i] / (p_c[i] + r_c[i]) if tp[i] > 0 else 0.0 for i in range(len(tp))]

        p_o = tp.sum().float() / (tp + fp).sum().float() * 100.0
        r_o = tp.sum().float() / (tp + fn).sum().float() * 100.0
        f_o = 2 * p_o * r_o / (p_o + r_o)

        if i % 100 == 0:
            print_log_message('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Precision {prec.val:.2f} ({prec.avg:.2f})\t'
                  'Recall {rec.val:.2f} ({rec.avg:.2f})'.format(
                epoch, i, len(data_loader), batch_time=batch_time,
                loss=losses, prec=prec, rec=rec))

    return f_o, losses.avg


def validate_multi(data_loader, model, criteria, device='cuda'):
    batch_time = AverageMeter()
    losses = AverageMeter()
    prec = AverageMeter()
    rec = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    tp, fp, fn, tn, count = 0, 0, 0, 0, 0
    tp_size, fn_size = 0, 0
    with torch.no_grad():
        for i, (input, target) in enumerate(data_loader):
            input = input.to(device=device)

            target = target.to(device=device)
            original_target = target
            target = target.max(dim=1)[0]

            # compute output
            output = model(input)
            loss = criteria(output, target.float())

            # measure accuracy and record loss
            pred = output.data.gt(0.0).long()

            tp += (pred + target).eq(2).sum(dim=0)
            fp += (pred - target).eq(1).sum(dim=0)
            fn += (pred - target).eq(-1).sum(dim=0)
            tn += (pred + target).eq(0).sum(dim=0)
            three_pred = pred.unsqueeze(1).expand(-1, 3, -1)  # n, 3, 80
            tp_size += (three_pred + original_target).eq(2).sum(dim=0)
            fn_size += (three_pred - original_target).eq(-1).sum(dim=0)
            count += input.size(0)

            this_tp = (pred + target).eq(2).sum()
            this_fp = (pred - target).eq(1).sum()
            this_fn = (pred - target).eq(-1).sum()
            this_tn = (pred + target).eq(0).sum()
            this_acc = (this_tp + this_tn).float() / (this_tp + this_tn + this_fp + this_fn).float()

            this_prec = this_tp.float() / (this_tp + this_fp).float() * 100.0 if this_tp + this_fp != 0 else 0.0
            this_rec = this_tp.float() / (this_tp + this_fn).float() * 100.0 if this_tp + this_fn != 0 else 0.0

            losses.update(float(loss), input.size(0))
            prec.update(float(this_prec), input.size(0))
            rec.update(float(this_rec), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            p_c = [float(tp[i].float() / (tp[i] + fp[i]).float()) * 100.0 if tp[i] > 0 else 0.0 for i in range(len(tp))]
            r_c = [float(tp[i].float() / (tp[i] + fn[i]).float()) * 100.0 if tp[i] > 0 else 0.0 for i in range(len(tp))]
            f_c = [2 * p_c[i] * r_c[i] / (p_c[i] + r_c[i]) if tp[i] > 0 else 0.0 for i in range(len(tp))]

            mean_p_c = sum(p_c) / len(p_c)
            mean_r_c = sum(r_c) / len(r_c)
            mean_f_c = sum(f_c) / len(f_c)

            p_o = tp.sum().float() / (tp + fp).sum().float() * 100.0
            r_o = tp.sum().float() / (tp + fn).sum().float() * 100.0
            f_o = 2 * p_o * r_o / (p_o + r_o)

            if i % 100 == 0:
                print_log_message('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Precision {prec.val:.2f} ({prec.avg:.2f})\t'
                      'Recall {rec.val:.2f} ({rec.avg:.2f})'.format(
                    i, len(data_loader), batch_time=batch_time, loss=losses,
                    prec=prec, rec=rec))
                print('P_C {:.2f} R_C {:.2f} F_C {:.2f} P_O {:.2f} R_O {:.2f} F_O {:.2f}'
                      .format(mean_p_c, mean_r_c, mean_f_c, p_o, r_o, f_o))

        print_info_message(' * P_C {:.2f} R_C {:.2f} F_C {:.2f} P_O {:.2f} R_O {:.2f} F_O {:.2f}'
              .format(mean_p_c, mean_r_c, mean_f_c, p_o, r_o, f_o))
        return f_o, losses.avg