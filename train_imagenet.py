# import argparse
# import os
# import shutil
# import time
# import random

# import torch
# import torch.nn as nn
# import torch.nn.parallel
# import torch.backends.cudnn as cudnn
# import torch.optim
# import torch.utils.data
# import torchvision.transforms as transforms
# import torchvision.datasets as datasets
# import torchvision.models as models
# from MODELS.model_resnet_cbam import *
# from PIL import ImageFile
# ImageFile.LOAD_TRUNCATED_IMAGES = True
# model_names = sorted(name for name in models.__dict__
#     if name.islower() and not name.startswith("__")
#     and callable(models.__dict__[name]))

# parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
# parser.add_argument('data', metavar='DIR',
#                     help='path to dataset')
# parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet',
#                     help='model architecture: ' +
#                         ' | '.join(model_names) +
#                         ' (default: resnet18)')
# parser.add_argument('--depth', default=50, type=int, metavar='D',
#                     help='model depth')
# parser.add_argument('--ngpu', default=4, type=int, metavar='G',
#                     help='number of gpus to use')
# parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
#                     help='number of data loading workers (default: 4)')
# parser.add_argument('--epochs', default=90, type=int, metavar='N',
#                     help='number of total epochs to run')
# parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
#                     help='manual epoch number (useful on restarts)')
# parser.add_argument('-b', '--batch-size', default=256, type=int,
#                     metavar='N', help='mini-batch size (default: 256)')
# parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
#                     metavar='LR', help='initial learning rate')
# parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
#                     help='momentum')
# parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
#                     metavar='W', help='weight decay (default: 1e-4)')
# parser.add_argument('--print-freq', '-p', default=10, type=int,
#                     metavar='N', help='print frequency (default: 10)')
# parser.add_argument('--resume', default='', type=str, metavar='PATH',
#                     help='path to latest checkpoint (default: none)')
# parser.add_argument("--seed", type=int, default=1234, metavar='BS', help='input batch size for training (default: 64)')
# parser.add_argument("--prefix", type=str, required=True, metavar='PFX', help='prefix for logging & checkpoint saving')
# parser.add_argument('--evaluate', dest='evaluate', action='store_true', help='evaluation only')
# parser.add_argument('--att-type', type=str, choices=['BAM', 'CBAM'], default=None)
# best_prec1 = 0

# if not os.path.exists('./checkpoints'):
#     os.mkdir('./checkpoints')

# def main():
#     global args, best_prec1
#     global viz, train_lot, test_lot
#     args = parser.parse_args()
#     print ("args", args)

#     torch.manual_seed(args.seed)
#     torch.cuda.manual_seed_all(args.seed)
#     random.seed(args.seed)

#     # create model
#     if args.arch == "resnet":
#         model = ResidualNet( 'ImageNet', args.depth, 1000, args.att_type )

#     # define loss function (criterion) and optimizer
#     criterion = nn.CrossEntropyLoss().cuda()

#     optimizer = torch.optim.SGD(model.parameters(), args.lr,
#                             momentum=args.momentum,
#                             weight_decay=args.weight_decay)
#     model = torch.nn.DataParallel(model, device_ids=list(range(args.ngpu)))
#     #model = torch.nn.DataParallel(model).cuda()
#     model = model.cuda()
#     print ("model")
#     print (model)

#     # get the number of model parameters
#     print('Number of model parameters: {}'.format(
#         sum([p.data.nelement() for p in model.parameters()])))

#     # optionally resume from a checkpoint
#     if args.resume:
#         if os.path.isfile(args.resume):
#             print("=> loading checkpoint '{}'".format(args.resume))
#             checkpoint = torch.load(args.resume)
#             args.start_epoch = checkpoint['epoch']
#             best_prec1 = checkpoint['best_prec1']
#             model.load_state_dict(checkpoint['state_dict'])
#             if 'optimizer' in checkpoint:
#                 optimizer.load_state_dict(checkpoint['optimizer'])
#             print("=> loaded checkpoint '{}' (epoch {})"
#                   .format(args.resume, checkpoint['epoch']))
#         else:
#             print("=> no checkpoint found at '{}'".format(args.resume))


#     cudnn.benchmark = True

#     # Data loading code
#     traindir = os.path.join(args.data, 'train')
#     valdir = os.path.join(args.data, 'val')
#     normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                  std=[0.229, 0.224, 0.225])

#     # import pdb
#     # pdb.set_trace()
#     val_loader = torch.utils.data.DataLoader(
#         datasets.ImageFolder(valdir, transforms.Compose([
#                 transforms.Scale(256),
#                 transforms.CenterCrop(224),
#                 transforms.ToTensor(),
#                 normalize,
#                 ])),
#             batch_size=args.batch_size, shuffle=False,
#            num_workers=args.workers, pin_memory=True)
#     if args.evaluate:
#         validate(val_loader, model, criterion, 0)
#         return

#     train_dataset = datasets.ImageFolder(
#         traindir,
#         transforms.Compose([
#             transforms.RandomSizedCrop(size0),
#             transforms.RandomHorizontalFlip(),
#             transforms.ToTensor(),
#             normalize,
#         ]))

#     train_sampler = None

#     train_loader = torch.utils.data.DataLoader(
#         train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
#         num_workers=args.workers, pin_memory=True, sampler=train_sampler)

#     for epoch in range(args.start_epoch, args.epochs):
#         adjust_learning_rate(optimizer, epoch)
        
#         # train for one epoch
#         train(train_loader, model, criterion, optimizer, epoch)
        
#         # evaluate on validation set
#         prec1 = validate(val_loader, model, criterion, epoch)
        
#         # remember best prec@1 and save checkpoint
#         is_best = prec1 > best_prec1
#         best_prec1 = max(prec1, best_prec1)
#         save_checkpoint({
#             'epoch': epoch + 1,
#             'arch': args.arch,
#             'state_dict': model.state_dict(),
#             'best_prec1': best_prec1,
#             'optimizer' : optimizer.state_dict(),
#         }, is_best, args.prefix)


# def train(train_loader, model, criterion, optimizer, epoch):
#     batch_time = AverageMeter()
#     data_time = AverageMeter()
#     losses = AverageMeter()
#     top1 = AverageMeter()
#     top5 = AverageMeter()

#     # switch to train mode
#     model.train()

#     end = time.time()
#     for i, (input, target) in enumerate(train_loader):
#         # measure data loading time
#         data_time.update(time.time() - end)
        
#         target = target.cuda(async=True)
#         input_var = torch.autograd.Variable(input)
#         target_var = torch.autograd.Variable(target)
        
#         # compute output
#         output = model(input_var)
#         loss = criterion(output, target_var)
        
#         # measure accuracy and record loss
#         prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
#         losses.update(loss.data[0], input.size(0))
#         top1.update(prec1[0], input.size(0))
#         top5.update(prec5[0], input.size(0))
        
#         # compute gradient and do SGD step
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
        
#         # measure elapsed time
#         batch_time.update(time.time() - end)
#         end = time.time()
        
#         if i % args.print_freq == 0:
#             print('Epoch: [{0}][{1}/{2}]\t'
#                   'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
#                   'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
#                   'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
#                   'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
#                   'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
#                    epoch, i, len(train_loader), batch_time=batch_time,
#                    data_time=data_time, loss=losses, top1=top1, top5=top5))

# def validate(val_loader, model, criterion, epoch):
#     batch_time = AverageMeter()
#     losses = AverageMeter()
#     top1 = AverageMeter()
#     top5 = AverageMeter()

#     # switch to evaluate mode
#     model.eval()

#     end = time.time()
#     for i, (input, target) in enumerate(val_loader):
#         target = target.cuda(async=True)
#         input_var = torch.autograd.Variable(input, volatile=True)
#         target_var = torch.autograd.Variable(target, volatile=True)
        
#         # compute output
#         output = model(input_var)
#         loss = criterion(output, target_var)
        
#         # measure accuracy and record loss
#         prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
#         losses.update(loss.data[0], input.size(0))
#         top1.update(prec1[0], input.size(0))
#         top5.update(prec5[0], input.size(0))
        
#         # measure elapsed time
#         batch_time.update(time.time() - end)
#         end = time.time()
        
#         if i % args.print_freq == 0:
#             print('Test: [{0}/{1}]\t'
#                   'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
#                   'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
#                   'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
#                   'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
#                    i, len(val_loader), batch_time=batch_time, loss=losses,
#                    top1=top1, top5=top5))
    
#     print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
#             .format(top1=top1, top5=top5))

#     return top1.avg


# def save_checkpoint(state, is_best, prefix):
#     filename='./checkpoints/%s_checkpoint.pth.tar'%prefix
#     torch.save(state, filename)
#     if is_best:
#         shutil.copyfile(filename, './checkpoints/%s_model_best.pth.tar'%prefix)


# class AverageMeter(object):
#     """Computes and stores the average and current value"""
#     def __init__(self):
#         self.reset()

#     def reset(self):
#         self.val = 0
#         self.avg = 0
#         self.sum = 0
#         self.count = 0

#     def update(self, val, n=1):
#         self.val = val
#         self.sum += val * n
#         self.count += n
#         self.avg = self.sum / self.count


# def adjust_learning_rate(optimizer, epoch):
#     """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
#     lr = args.lr * (0.1 ** (epoch // 30))
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr


# def accuracy(output, target, topk=(1,)):
#     """Computes the precision@k for the specified values of k"""
#     maxk = max(topk)
#     batch_size = target.size(0)

#     _, pred = output.topk(maxk, 1, True, True)
#     pred = pred.t()
#     correct = pred.eq(target.view(1, -1).expand_as(pred))

#     res = []
#     for k in topk:
#         correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
#         res.append(correct_k.mul_(100.0 / batch_size))
#     return res


# if __name__ == '__main__':
#     main()


import argparse
import os
import shutil
import time
import random

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from MODELS.model_resnet_cbam import *
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch Image Classification Training')
parser.add_argument('data', metavar='DIR', help='path to dataset')
parser.add_argument('--dataset', type=str, choices=['CIFAR10', 'CIFAR100', 'TinyImageNet'], default='CIFAR10')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet', help='model architecture')
parser.add_argument('--depth', default=50, type=int, metavar='D', help='model depth')
parser.add_argument('--ngpu', default=1, type=int, metavar='G', help='number of gpus to use')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='data loading workers')
parser.add_argument('--epochs', default=90, type=int, metavar='N', help='total epochs')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number')
parser.add_argument('-b', '--batch-size', default=128, type=int, metavar='N', help='mini-batch size')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W', help='weight decay')
parser.add_argument('--print-freq', '-p', default=10, type=int, metavar='N', help='print frequency')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='checkpoint path')
parser.add_argument('--seed', default=1234, type=int, metavar='S', help='random seed')
parser.add_argument('--prefix', type=str, required=True, metavar='PFX', help='logging & checkpoint prefix')
parser.add_argument('--evaluate', dest='evaluate', action='store_true', help='only evaluate')
parser.add_argument('--att-type', type=str, choices=['BAM', 'CBAM'], default=None)

best_prec1 = 0

if not os.path.exists('./checkpoints'):
    os.mkdir('./checkpoints')

import requests
import zipfile

def download_tinyimagenet(data_dir):
    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    os.makedirs(data_dir, exist_ok=True)
    zip_path = os.path.join(data_dir, "tiny-imagenet-200.zip")
    
    if not os.path.exists(os.path.join(data_dir, "tiny-imagenet-200")):
        print("Downloading TinyImageNet...")
        response = requests.get(url, stream=True)
        with open(zip_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Extracting TinyImageNet...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(data_dir)
        os.remove(zip_path)
        print("TinyImageNet downloaded and extracted.")

def main():
    global args, best_prec1
    args = parser.parse_args()
    print("args", args)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)

    if args.dataset == 'CIFAR10':
        num_classes = 10
        input_size = 32
        dataset_type = datasets.CIFAR10
    elif args.dataset == 'CIFAR100':
        num_classes = 100
        input_size = 32
        dataset_type = datasets.CIFAR100
    elif args.dataset == 'TinyImageNet':
        num_classes = 200
        input_size = 64
        dataset_type = None

    model = ResidualNet(args.dataset, args.depth, num_classes, args.att_type)

    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    model = torch.nn.DataParallel(model, device_ids=list(range(args.ngpu)))
    model = model.cuda()

    print("Number of model parameters: {}".format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

    if args.resume and os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))

    cudnn.benchmark = True

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    if args.dataset in ['CIFAR10', 'CIFAR100']:
        train_dataset = dataset_type(root=args.data, train=True, download=True,
            transform=transforms.Compose([
                transforms.RandomCrop(input_size, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))

        val_dataset = dataset_type(root=args.data, train=False, download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ]))
    elif args.dataset == 'TinyImageNet':
        download_tinyimagenet(args.data)
        train_dataset = datasets.ImageFolder(
            os.path.join(args.data, "tiny-imagenet-200", "train"),
            transform=transforms.Compose([
                transforms.RandomResizedCrop(input_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))
        val_dataset = datasets.ImageFolder(
            os.path.join(args.data, "tiny-imagenet-200", "val"),
            transform=transforms.Compose([
                transforms.Resize((input_size, input_size)),
                transforms.ToTensor(),
                normalize,
            ]))
    else:
        train_dataset = datasets.ImageFolder(os.path.join(args.data, 'train'),
            transform=transforms.Compose([
                transforms.RandomResizedCrop(input_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))

        val_dataset = datasets.ImageFolder(os.path.join(args.data, 'val'),
            transform=transforms.Compose([
                transforms.Resize((input_size, input_size)),
                transforms.ToTensor(),
                normalize,
            ]))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        validate(val_loader, model, criterion, 0)
        return

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)
        train(train_loader, model, criterion, optimizer, epoch)
        prec1 = validate(val_loader, model, criterion, epoch)

        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict(),
        }, is_best, args.prefix)

def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        data_time.update(time.time() - end)
        target = target.cuda(non_blocking=True)

        output = model(input)
        loss = criterion(output, target)

        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]	'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})	'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})	'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})	'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})	'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))

def validate(val_loader, model, criterion, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda(non_blocking=True)
            output = model(input)
            loss = criterion(output, target)

            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]	'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})	'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})	'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})	'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       i, len(val_loader), batch_time=batch_time, loss=losses,
                       top1=top1, top5=top5))

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg

def save_checkpoint(state, is_best, prefix):
    filename = f'./checkpoints/{prefix}_checkpoint.pth.tar'
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, f'./checkpoints/{prefix}_model_best.pth.tar')

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def adjust_learning_rate(optimizer, epoch):
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

if __name__ == '__main__':
    main()
