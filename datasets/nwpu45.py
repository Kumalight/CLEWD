import torch
import torchvision.transforms as transforms
import os.path as osp
from .folder import ImageFolder


def form_nwpu45(config):

    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    nclasses = 45

    source_train_root = osp.join(config.dataroot, config.dataset, 'train')
    target_root = osp.join(config.dataroot, config.dataset, 'test')

    #transform_source = transforms.Compose([transforms.RandomResizedCrop(224),transforms.RandomHorizontalFlip(),transforms.ToTensor(),transforms.Normalize(mean=mean, std=std)] )
    transform_source  = transforms.Compose([ transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),

        transforms.RandomRotation(degrees=(-180, 180)),
        transforms.ToTensor(), transforms.Normalize(mean=mean, std=std),
    ])




    #transform_target = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224),transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)] )

    transform_target = transforms.Compose([transforms.Resize(256),
        transforms.CenterCrop(224),transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])



    source_d = ImageFolder(root=source_train_root, transform=transform_source)
    target_d = ImageFolder(root=target_root, transform=transform_target)

    source_loader = torch.utils.data.DataLoader(source_d, batch_size=config.batchSize,
                                                shuffle=True, num_workers=4, pin_memory=True)
    target_loader = torch.utils.data.DataLoader(target_d, batch_size=config.batchSize,
                                                shuffle=False, num_workers=4, pin_memory=True)

    return source_loader, target_loader, nclasses
