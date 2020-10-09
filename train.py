import argparse
import numpy as np
import torch
import torch.optim as optim
import collections
import time
import math
import datetime
from torch.utils.data import DataLoader

from models.retinanet import build_retinanet
from dataset.voc_dataset import VOCDataset, AnnotationTransform, detection_collate
from dataset.data_augment import PreProcess

def main(args=None):
    parser = argparse.ArgumentParser(description='Rotate RetinaNet.')

    parser.add_argument('--dataset_path', help='Path to COCO directory',
                        default='/home/fengkai/datasets/UCAS-AOD/')
    parser.add_argument('--epochs', help='Number of epochs', type=int, default=50)
    parser.add_argument('--batch_size', help='Number of batch size', type=int, default=2)
    parser.add_argument('--num_workers', help='Number of num_workers', type=int, default=8)
    parser.add_argument('--num_class', help='Number of num_class', type=int, default=2)
    parser.add_argument('--use_gpu', help='Use gpu', type=bool, default=True)
    parser.add_argument('--ngpu', help='gpus', type=int, default=1)

    parser = parser.parse_args(args)

    dataset = VOCDataset(parser.dataset_path, preprocess=PreProcess(), target_transform=AnnotationTransform())
    dataloader = DataLoader(dataset, batch_size=parser.batch_size,
                            shuffle=True, num_workers=parser.num_workers, collate_fn=detection_collate)

    retinanet = build_retinanet(num_classes=parser.num_class, pretrained=True)
    if parser.use_gpu:
        if torch.cuda.is_available():
            retinanet = retinanet.cuda()

    if parser.use_gpu and parser.ngpu > 1:
        retinanet = torch.nn.DataParallel(retinanet, device_ids=list(range(parser.ngpu))).cuda()

    retinanet.training = True

    optimizer = optim.Adam(retinanet.parameters(), lr=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)

    loss_hist = collections.deque(maxlen=500)

    retinanet.train()
    retinanet.freeze_bn()
    print('Num training images: {}'.format(len(dataset)))
    
    epoch_size = math.ceil(len(dataset) / parser.batch_size)
    max_iter = parser.epochs * epoch_size
    
    for epoch_num in range(parser.epochs):
        retinanet.train()
        retinanet.freeze_bn()

        epoch_loss = []

        for iter, data in enumerate(dataloader):
            time_start = time.time()
            optimizer.zero_grad()

            if torch.cuda.is_available():
                images, targets = data
                images = images.cuda()
                targets = [t.cuda() for t in targets]
                classification_loss, regression_loss = retinanet([images, targets])
            else:
                images, targets = data
                classification_loss, regression_loss = retinanet([images, targets])
            classification_loss = classification_loss.mean()
            regression_loss = regression_loss.mean()

            loss = classification_loss + regression_loss

            if bool(loss == 0):
                continue

            loss.backward()

            torch.nn.utils.clip_grad_norm_(retinanet.parameters(), 0.1)

            optimizer.step()

            loss_hist.append(float(loss))

            epoch_loss.append(float(loss))

            time_end = time.time()
            batch_time = time_end - time_start
            eta = int(batch_time * (max_iter - (iter + epoch_num * epoch_size)))

            print(
                'Epoch: {} | Iteration: {} | Classification loss: {:1.5f} | Regression loss: {:1.5f} | Running loss: {:1.5f} | ETA: {}'.format(
                    epoch_num, iter, float(classification_loss), float(regression_loss), np.mean(loss_hist), str(datetime.timedelta(seconds=eta))))

            del classification_loss
            del regression_loss

        scheduler.step(np.mean(epoch_loss))
        torch.save({'state_dict': retinanet.state_dict()}, 'rotate_retinanet_{}.pth'.format(epoch_num))

    retinanet.eval()
    torch.save({'state_dict': retinanet.state_dict()}, 'model_final.pth')

if __name__ == '__main__':
    main()



