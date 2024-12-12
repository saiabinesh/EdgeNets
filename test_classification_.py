import argparse
import torch
from utilities.utils import model_parameters, compute_flops
from utilities.train_eval_classification import validate
import os
from data_loader.classification.imagenet import test_loader as loader
from utilities.print_utils import *
#============================================
__author__ = "Sachin Mehta"
__license__ = "MIT"
__maintainer__ = "Sachin Mehta"
#============================================
from torch import nn

def main(args):
    # create model
    if args.model == 'dicenet':
        from model.classification import dicenet as net
        model = net.CNNModel(args)
    elif args.model == 'espnetv2':
        from model.classification import espnetv2 as net
        model = net.EESPNet(args)
    elif args.model == 'shufflenetv2':
        from model.classification import shufflenetv2 as net
        model = net.CNNModel(args)
    else:
        NotImplementedError('Model {} not yet implemented'.format(args.model))
        exit()

    num_params = model_parameters(model)
    flops = compute_flops(model)
    print_info_message('FLOPs: {:.2f} million'.format(flops))
    print_info_message('Network Parameters: {:.2f} million'.format(num_params))


    if not args.weights:
        print_info_message('Grabbing location of the ImageNet weights from the weight dictionary')
        from model.weight_locations.classification import model_weight_map

        weight_file_key = '{}_{}'.format(args.model, args.s)
        assert weight_file_key in model_weight_map.keys(), '{} does not exist'.format(weight_file_key)
        args.weights = model_weight_map[weight_file_key]

    num_gpus = torch.cuda.device_count()
    device = 'cuda' if num_gpus >=1 else 'cpu'
    weight_dict = torch.load(args.weights, map_location=torch.device(device))
    model.load_state_dict(weight_dict['state_dict']) #(weight_dict)

    if num_gpus >= 1:
        model = torch.nn.DataParallel(model)
        model = model.cuda()
        if torch.backends.cudnn.is_available():
            import torch.backends.cudnn as cudnn
            cudnn.benchmark = True
            cudnn.deterministic = True
    if args.dataset=='imagenet':
        # Data loading code
        val_loader = loader(args) #actually it's only loading test set now
        from utilities.train_eval_classification import validate
        validate(val_loader, model, criteria=None, device=device)
    elif args.dataset=='coco':
        from data_loader.classification.coco import COCOClassification
        # train_dataset = COCOClassification(root=args.data, split='train', year='2017', inp_size=args.inpSize,
                                           # scale=args.scale, is_training=True)
        val_dataset = COCOClassification(root=args.data, split='val', year='2017', inp_size=args.inpSize,
                                         is_training=False)

        # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                                   # pin_memory=True, num_workers=args.workers)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                                 pin_memory=True, num_workers=args.workers)
                                                 
        # print("size(train_loader) ",size(train_loader))
        # print("size(val_loader) ",size(val_loader))

        criterion = nn.BCEWithLogitsLoss()
        acc_metric = 'F1'                                        

        # # import the loaders too
        # from utilities.train_eval_classification import train_multi as train
        from utilities.train_eval_classification import validate_multi as validate
        validate(val_loader, model, criteria=criterion, device=device)


if __name__ == '__main__':
    from commons.general_details import classification_models, classification_datasets

    parser = argparse.ArgumentParser(description='Testing efficient networks')
    parser.add_argument('--workers', default=4, type=int, help='number of data loading workers (default: 4)')
    parser.add_argument('--data', default='', help='path to dataset')
    parser.add_argument('--dataset', default='imagenet', help='Name of the dataset', choices=classification_datasets)
    parser.add_argument('--batch-size', default=512, type=int, help='mini-batch size (default: 512)')
    parser.add_argument('--num-classes', default=1000, type=int, help='# of classes in the dataset')
    parser.add_argument('--s', default=1, type=float, help='Width scaling factor')
    parser.add_argument('--weights', type=str, default='', help='weight file')
    parser.add_argument('--inpSize', default=224, type=int, help='Input size')
    ##Select a model
    parser.add_argument('--model', default='dicenet', choices=classification_models, help='Which model?')
    parser.add_argument('--model-width', default=224, type=int, help='Model width')
    parser.add_argument('--model-height', default=224, type=int, help='Model height')
    parser.add_argument('--channels', default=3, type=int, help='Input channels')

    args = parser.parse_args()
    main(args)
