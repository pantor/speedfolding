from pathlib import Path
from argparse import ArgumentParser
from collections import defaultdict
import time

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import BiMamaNet
from dataset import BiFoldingDataset


class Trainer:
    def __init__(self, collection: str, checkpoint_path: Path = None, experiment_name: str = None, num_workers=4):
        train_action, test_actions = BiFoldingDataset.load_train_test_actions(collection)
    
        train_dataset = BiFoldingDataset(collection, train_action, use_augmentation=True)
        self.train_data = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=num_workers)

        test_dataset = BiFoldingDataset(collection, test_actions)
        self.test_data = DataLoader(test_dataset, batch_size=8, shuffle=True, num_workers=num_workers)

        self.checkpoint_path = checkpoint_path
        self.experiment_name = experiment_name if experiment_name else ''

        if experiment_name:
            self.checkpoint_path = self.checkpoint_path / experiment_name
        
        self.checkpoint_path.mkdir(parents=True, exist_ok=True)

    def fit(self, model, epochs: int):
        optimizer = optim.Adam(model.parameters(), lr=4e-4, weight_decay=1e-6)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.97)

        writer = SummaryWriter(comment=self.experiment_name)

        min_test_loss = 1e12
        best_checkpoint_path, best_epoch = None, 0
        for epoch in range(epochs):
            train_metrics = defaultdict(float)
            model.train()
            
            if args.type == 'embedding':
                model.offset = max(2.0 - 0.2 * epoch, 0.0)
            
            for i_batch, batch in enumerate(self.train_data):
                m = model.forward_loss(batch)
                optimizer.zero_grad()
                m['loss'].backward()
                optimizer.step()

                for k, v in m.items():
                    train_metrics[k] += v.item()

                print('[%d/%d, %5d] loss %0.3f, fling: %.3f, pnh: %0.3f' % (epoch + 1, epochs, i_batch + 1, m['loss'].item(), m['loss_fling'].item(), m['loss_picknhold'].item()), end='')
                print('\r', end='')

            for k in train_metrics.keys():
                train_metrics[k] /= i_batch + 1
            
            print(f'[{epoch+1}/{epochs}] train loss: ' + ' | '.join(f'{k} {v:0.4f}' for k, v in train_metrics.items()))
            for k, v in train_metrics.items():
                writer.add_scalar(f'{k}/train', v, epoch)
            
            test_metrics = defaultdict(float)
            model.eval()
            with torch.no_grad():
                for i_batch, batch in enumerate(self.test_data):
                    m = model.forward_loss(batch)
                    for k, v in m.items():
                        test_metrics[k] += v.item()
            
            for k in test_metrics.keys():
                test_metrics[k] /= i_batch + 1

            scheduler.step()

            print(f'[{epoch+1}/{epochs}] test loss:  ' + ' | '.join(f'{k} {v:0.4f}' for k, v in test_metrics.items()))
            for k, v in test_metrics.items():
                writer.add_scalar(f'{k}/test', v, epoch)
            
            if test_metrics['loss'] < min_test_loss:
                min_test_loss = test_metrics['loss']
                best_checkpoint_path = self.checkpoint_path / f'ckpt.pth'

                torch.save(model.state_dict(), best_checkpoint_path)
                print(f'[{epoch+1}/{epochs}] improves test loss')
                best_epoch = epoch

        print(f'Saved best checkpoint from epoch {best_epoch} to {best_checkpoint_path}')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-e', dest='exp_name', type=str, default=None, help='defines the experiment name and where the checkpoints are saved')
    parser.add_argument('-t', dest='type', type=str, default=None, help='type of the model, e.g. conditioned')
    
    args = parser.parse_args()

    print('Training', args.type)

    if args.exp_name:
        args.exp_name = args.exp_name + '_' + time.strftime('%Y%m%d_%H%M%S')
    
    model = BiMamaNet(20)
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        model = model.cuda()

    save_path = Path().parent / 'checkpoints'
    trainer = Trainer('test', checkpoint_path=save_path, experiment_name=args.exp_name)
    trainer.fit(model, epochs=100)
