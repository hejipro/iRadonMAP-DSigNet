'''
Description: 
Author: heji
Date: 2021-04-26 14:46:35
LastEditTime: 2021-04-29 21:41:36
LastEditors: GuoYi
'''

import pickle
import torch
import time 
import os
from torch.autograd import Variable 
import scipy.io


def train_model(dataloaders, model, optimizer, criterion=None, scheduler=None, min_loss=None, pre_losses=None, num_epochs=25, dataset_sizes=None, opt=None):
    since = time.time()

    if min_loss is None:
        min_loss = {x:1.0 for x in ['train', 'val']}
        
    pre_epoch = 0
    losses = {x: torch.zeros(0, opt.batch_num[x]) for x in ['train', 'val']}
    
    if pre_losses is not None:
        pre_epoch = losses['train'].size(0)
        min_dim = {x: min(opt.batch_num[x], pre_losses[x].size(1)) for x in ['train', 'val']}
        losses = {x: pre_losses[x][:,:min_dim[x]] for x in ['train', 'val']}

    epoch_loss = {x: 0.0 for x in ['train', 'val']}

    for epoch in range(pre_epoch, num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase is 'train':
                if scheduler is not None:
                    scheduler.step()
                model.train()  # Set model to training mode
            elif phase is 'val':
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            tmp_losses = torch.zeros(1, 0)
            
            # Iterate over data.
            for i_batch, data in enumerate(dataloaders[phase]):
                if i_batch == opt.batch_num[phase]:
                    break

                labels = data['ndct']
                inputs = data['sinogram']
                if opt.use_cuda:  # wrap them in Variable
                    labels = Variable(labels).cuda(opt.gpu_id_conv)
                    inputs = Variable(inputs).cuda(opt.gpu_id_conv)
                else:
                    labels = Variable(labels)
                    inputs = Variable(inputs)
                    
                if phase is 'train':
                    optimizer.zero_grad()
                
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()

                    optimizer.step()

                elif phase is 'val':
                    with torch.no_grad():
                        outputs = model(inputs)
                    loss = criterion(outputs, labels)

                print('{}, {}, {}, {}/{}, subLoss: {:.8f}'.format(opt.net_name, phase, epoch, i_batch, opt.batch_num[phase], loss.data.item()))
                # statistics
                tmp_losses = torch.cat((tmp_losses, torch.tensor([[loss.data.item()]])), 1)
                
                running_loss += loss.data.item() * inputs.size(0)

            epoch_loss[phase] = running_loss / (dataset_sizes[phase])

            print('{} Loss: {:.8f}'.format(phase, epoch_loss[phase]))
            
            if phase is 'val':
                print('Train / Val: {:.8f}'.format(epoch_loss['train']/epoch_loss['val']))

            losses[phase] = torch.cat((losses[phase], tmp_losses[:,:min_dim[phase]]), 0)
            
            #deep copy the model
            if epoch_loss[phase] < min_loss[phase]:
                min_loss[phase] = epoch_loss[phase]
                min_loss_mat = {key: value for key, value in min_loss.items()}
                scipy.io.savemat(opt.root_path + 'Loss_save/min_loss_{}.mat'.format(opt.net_name), mdict = min_loss_mat)
                torch.save(model.state_dict(), 
                        opt.root_path + 'Model_save/best_{}_model_{}.pkl'.format(phase, opt.net_name))
                torch.save(optimizer.state_dict(),
                        opt.root_path + 'Optimizer_save/best_{}_optimizer_{}.pkl'.format(phase, opt.net_name))

        torch.save(model.state_dict(), 
                opt.root_path + 'Model_save/backup_model_{}.pkl'.format(opt.net_name))
        torch.save(optimizer.state_dict(),
                opt.root_path + 'Optimizer_save/backup_optimizer_{}.pkl'.format(opt.net_name))
                
        print()
        
        losses_mat = {key: value.numpy() for key, value in losses.items()}
        scipy.io.savemat(opt.root_path + 'Loss_save/losses_{}.mat'.format(opt.net_name), mdict = losses_mat)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Minimun train loss: {:5f}'.format(min_loss['train']))

    