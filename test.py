import argparse

import evaluation
import yaml
import torch
import utils
import keras
from keras.models import Model
from keras.layers import Input,Dense,PReLU,Dropout #PRelU为带参数的ReLU


def main(opt, current_config):
    model_checkpoint = opt.checkpoint

    checkpoint = torch.load('runs/runX/checkpointHtk32.pth.tar')
    print('Checkpoint loaded from {}'.format(model_checkpoint))
    loaded_config = checkpoint['config']
 #   capemb = torch.load('runs/runX/ranks.pth.tar')
   # print(capemb.shape())
    #checkpoint =dict(checkpoint,capemb)
    #checkpoint = merge_model(checkpoint,capemb)

    if opt.size == "1k":
        fold5 = True
    elif opt.size == "5k":
        fold5 = False
    else:
        raise ValueError('Test split size not recognized!')

    # Override some mandatory things in the configuration (paths)
    if current_config is not None:
        loaded_config['dataset']['images-path'] = current_config['dataset']['images-path']
        loaded_config['dataset']['data'] = current_config['dataset']['data']
        loaded_config['image-model']['pre-extracted-features-root'] = current_config['image-model']['pre-extracted-features-root']

    evaluation.evalrank(loaded_config, checkpoint, split="test",opt=opt,fold5=fold5)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, help="Checkpoint to load")
    parser.add_argument('--size', type=str, choices=['1k', '5k'], default='1k')
    parser.add_argument('--config', type=str, default='configs/hosa_coco_MrSw.yaml', help="Which configuration to use for overriding the checkpoint configuration. See into 'config' folder")
    parser.add_argument('--workers', default=1, type=int,help='Number of data loader workers.')

    ##
    #
    # parser.add_argument('--num_epochs', default=1, type=int,
    #                    help='Number of training epochs.')
    # # parser.add_argument('--crop_size', default=224, type=int,
    # #                     help='Size of an image crop as the CNN input.')
    # parser.add_argument('--lr_update', default=15, type=int,
    #                     help='Number of epochs to update the learning rate.')
    # parser.add_argument('--log_step', default=10, type=int,
    #                     help='Number of steps to print and record the log.')
    # parser.add_argument('--val_step', default=500, type=int,
    #                     help='Number of steps to run validation.')
    # parser.add_argument('--test_step', default=100000000, type=int,
    #                     help='Number of steps to run validation.')
    # parser.add_argument('--logger_name', default='runs/runX',
    #                     help='Path to save the model and Tensorboard log.')
    # parser.add_argument('--resume', default='', type=str, metavar='PATH',
    #                     help='path to latest checkpoint (default: none). Loads model, optimizer, scheduler')
    # parser.add_argument('--load-model', default='', type=str, metavar='PATH',
    #                     help='path to latest checkpoint (default: none). Loads only the model')
    # parser.add_argument('--use_restval', action='store_true',
    #                     help='Use the restval data for training on MSCOCO.')
    # parser.add_argument('--reinitialize-scheduler', action='store_true',
    #                     help='Reinitialize scheduler. To use with --resume')

    ##
    opt = parser.parse_args()

    if opt.config is not None:
        with open(opt.config, 'r') as ymlfile:
            config = yaml.load(ymlfile,Loader=yaml.FullLoader)

    else:
        config = None
    main(opt, config)