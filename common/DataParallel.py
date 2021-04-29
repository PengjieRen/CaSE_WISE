import os
import torch
from torch.utils.data import DistributedSampler
import torch.nn.init
from torch.cuda.amp import *
import time
import sys

class DataParallel(object):
    def __init__(self, model, optimizer, scheduler,  local_rank):
        super(DataParallel, self).__init__()
        self.local_rank = local_rank
        self.optimizer = optimizer
        self.scheduler = scheduler

        if local_rank is not None and torch.cuda.is_available():
            torch.cuda.set_device(local_rank)

        if torch.cuda.is_available():
            self.model = model.cuda()
        else:
            self.model = model

        if torch.cuda.is_available() and local_rank is not None:
            self.model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    def serialize(self, epoch, path):
        if torch.cuda.is_available() and self.local_rank != 0:
            return
        if not os.path.exists(path):
            os.makedirs(path)
        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            torch.save(self.model.module.state_dict(), os.path.join(path, '.'.join([str(epoch), 'model'])))
        else:
            torch.save(self.model.state_dict(), os.path.join(path, '.'.join([str(epoch), 'model'])))

    def train_epoch(self, method, train_dataset, train_collate_fn, batch_size, epoch):
        self.model.train()
        if torch.cuda.is_available():
            sampler = DistributedSampler(train_dataset)
            sampler.set_epoch(epoch)
            train_loader = torch.utils.data.DataLoader(train_dataset, collate_fn=train_collate_fn, batch_size=batch_size, sampler=sampler, pin_memory=True, num_workers=0)
        else:
            train_loader = torch.utils.data.DataLoader(train_dataset, collate_fn=train_collate_fn, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=0)

        start_time = time.time()
        losses = dict()
        count = 0
        for j, data in enumerate(train_loader, 0):
            if torch.cuda.is_available():
                data_cuda = dict()
                for key, value in data.items():
                    if isinstance(value, torch.Tensor):
                        data_cuda[key] = value.cuda()
                    else:
                        data_cuda[key] = value
                data = data_cuda

            loss = self.model(data, method)
            sum_loss = torch.cat([loss[k].mean().reshape(1) for k in loss]).sum()

            sum_loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)

            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()
            self.optimizer.zero_grad()

            count+=1
            if count%(100)==0:
                elapsed_time = time.time() - start_time
                closs = dict()
                for l in loss:
                    closs[l] = loss[l].cpu().item()
                output = {'Epoch': epoch, 'batch': count, 'Loss': closs, 'Time': elapsed_time, 'Leaning rate': self.scheduler.get_last_lr()}
                print(output)
                sys.stdout.flush()

            if 'id' not in losses:
                losses['id']=data['id'].cpu()
            else:
                losses['id']=torch.cat([losses['id'], data['id'].cpu()], dim=0)
            for k in loss:
                v = loss[k].reshape(1)
                if k not in losses:
                    losses[k] = v.cpu()
                else:
                    losses[k] = torch.cat([losses[k], v.cpu()], dim=0)


        return losses

    def test_epoch(self, method, test_dataset, test_collate_fn, batch_size):
        self.model.eval()
        with torch.no_grad():
            if torch.cuda.is_available():
                sampler = DistributedSampler(test_dataset, shuffle=False)
                test_loader = torch.utils.data.DataLoader(test_dataset, collate_fn=test_collate_fn, batch_size=batch_size, sampler=sampler, pin_memory=True, num_workers=0)
            else:
                test_loader = torch.utils.data.DataLoader(test_dataset, collate_fn=test_collate_fn, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=0)

            outputs = dict()
            for k, data in enumerate(test_loader, 0):
                if torch.cuda.is_available():
                    data_cuda = dict()
                    for key, value in data.items():
                        if isinstance(value, torch.Tensor):
                            data_cuda[key] = value.cuda()
                        else:
                            data_cuda[key] = value
                    data = data_cuda

                output = self.model(data, method)

                if 'id' not in outputs:
                    outputs['id'] = data['id'].cpu()
                else:
                    outputs['id'] = torch.cat([outputs['id'], data['id'].cpu()], dim=0)
                for k in output:
                    v = output[k]
                    if k not in outputs:
                        outputs[k] = v.cpu()
                    else:
                        outputs[k] = torch.cat([outputs[k], v.cpu()], dim=0)

        return outputs