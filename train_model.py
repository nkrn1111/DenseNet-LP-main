import fire
import os
import time
import torch
import numpy as np
import networkx as nx
from models import DenseNet
import PreProcessingFunctions as PFunctions


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"

#this is for clearing cuda memorry
torch.cuda.empty_cache()

# read number of nodes in a graph
num_nodes_file = open(PFunctions.num_of_nodes_file_name, 'r')

lines = num_nodes_file.readlines()
str_num = ""
for line in lines:
    for c in line:
        if c.isdigit() == True:
            str_num = str_num + c
       
num_nodes_file.close()
num_of_nodes = int(str_num) 
print("num_of_nodes = ", num_of_nodes)

num_of_node_pairs = int((num_of_nodes*(num_of_nodes - 1))/2)
batch_size_divider = round(num_of_node_pairs/PFunctions.batch_size)

#load training labels
load_train_labels = PFunctions.load_data_from_file(PFunctions.train_labels_file_name)
training_labels = np.array_split(load_train_labels, batch_size_divider)

#load test labels
load_test_labels = PFunctions.load_data_from_file(PFunctions.test_labels_file_name)
testing_labels = np.array_split(load_test_labels, batch_size_divider)

class AverageMeter(object):
    """
    Computes and stores the average and current value
    Copied from: https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
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


def train_epoch(model, train_batch_list, optimizer, epoch, n_epochs, print_freq=1):
    batch_time = AverageMeter()
    losses = AverageMeter()
    error = AverageMeter()

    # Model on train mode
    model.train()

    end = time.time()
    for batch_idx , batch in enumerate(train_batch_list):

        torchLabels = torch.cuda.LongTensor(training_labels[batch_idx])
        torchMatrix = torch.cuda.FloatTensor(batch)
        input = torchMatrix
        target = torchLabels

        # compute output
        output = model(input)
        loss = torch.nn.functional.cross_entropy(output, target)

        # measure accuracy and record loss
        batch_size = target.size(0)
        _, pred = output.data.cpu().topk(1, dim=1)
        error.update(torch.ne(pred.squeeze(), target.cpu()).float().sum().item() / batch_size, batch_size)
        losses.update(loss.item(), batch_size)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

       
        # print stats
        if batch_idx % print_freq == 0:
            res = '\t'.join([
                'Epoch: [%d/%d]' % (epoch + 1, n_epochs),
                'Iter: [%d/%d]' % (batch_idx + 1, len(train_batch_list)),
                'Time %.3f (%.3f)' % (batch_time.val, batch_time.avg),
                'Loss %.4f (%.4f)' % (losses.val, losses.avg),
                'Error %.4f (%.4f)' % (error.val, error.avg),
            ])
            print(res)

    # Return summary statistics
    return batch_time.avg, losses.avg, error.avg


def test_epoch(model, test_batch_list, print_freq=1, is_test=True):
    batch_time = AverageMeter()
    losses = AverageMeter()
    error = AverageMeter()

    # Model on eval mode
    model.eval()

    end = time.time()
   
    for batch_idx , batch in enumerate(test_batch_list):
     
        torchLabels = torch.cuda.LongTensor(testing_labels[batch_idx])   
        torchMatrix = torch.cuda.FloatTensor(batch)
        input = torchMatrix
        target = torchLabels
        
        # compute output
        output = model(input)
        loss = torch.nn.functional.cross_entropy(output, target)

        # measure accuracy and record loss
        batch_size = target.size(0)
        _, pred = output.data.cpu().topk(1, dim=1)
        error.update(torch.ne(pred.squeeze(), target.cpu()).float().sum().item() / batch_size, batch_size)
        losses.update(loss.item(), batch_size)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print stats
        if batch_idx % print_freq == 0:
            res = '\t'.join([
                'Test' if is_test else 'Valid',
                'Iter: [%d/%d]' % (batch_idx + 1, len(test_batch_list)),
                'Time %.3f (%.3f)' % (batch_time.val, batch_time.avg),
                'Loss %.4f (%.4f)' % (losses.val, losses.avg),
                'Error %.4f (%.4f)' % (error.val, error.avg),
            ])
            print(res)

    # Return summary statistics
    return batch_time.avg, losses.avg, error.avg


def train(model, test_batch_list, train_batch_list, save, n_epochs=300 ,
           lr=0.1, wd=0.0001, momentum=0.9, seed=None):
    if seed is not None:
        torch.manual_seed(seed)


    # Model on cuda
    if torch.cuda.is_available():
        model = model.cuda()

    # Wrap model for multi-GPUs, if necessary
    model_wrapper = model
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        model_wrapper = torch.nn.DataParallel(model).cuda()

    # Optimizer
    optimizer = torch.optim.SGD(model_wrapper.parameters(), lr=lr, momentum=momentum, nesterov=True, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[0.5 * n_epochs, 0.75 * n_epochs],
                                                     gamma=0.1)

    # Start log
    with open(os.path.join(save, 'results.csv'), 'w') as f:
        f.write('epoch,train_loss,train_error,valid_loss,valid_error,test_error\n')

    # Train model
    
    for epoch in range(n_epochs):
        _, train_loss, train_error = train_epoch(
            model=model_wrapper,
            train_batch_list=train_batch_list,
            optimizer=optimizer,
            epoch=epoch,
            n_epochs=n_epochs,
        )
        scheduler.step()
        _, valid_loss, valid_error = test_epoch(
            model=model_wrapper,
            test_batch_list=test_batch_list,
            is_test= True
        )

        torch.save(model.state_dict(), os.path.join(save, 'model.dat'))

        # Log results
        with open(os.path.join(save, 'results.csv'), 'a') as f:
            f.write('%03d,%0.6f,%0.6f,%0.5f,%0.5f,\n' % (
                (epoch + 1),
                train_loss,
                train_error,
                valid_loss,
                valid_error,
            ))

    # Final test of model on test set
    model.load_state_dict(torch.load(os.path.join(save, 'model.dat')))
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model).cuda()
    test_results = test_epoch(
        model=model,
        test_batch_list=test_batch_list,
        is_test=True
    )
    _, _, test_error = test_results
    with open(os.path.join(save, 'results.csv'), 'a') as f:
        f.write(',,,,,%0.5f\n' % (test_error))
    print('Final test error: %.4f' % test_error)


def start_training(depth=100, growth_rate=12, efficient=True,
         n_epochs=300, seed=None):
    """
    training of a Denset using graph node pairs data converted into pictures data

    Args:
        
        depth (int) - depth of the network (number of convolution layers) (default 40)
        growth_rate (int) - number of features added per DenseNet layer (default 12)
        efficient (bool) - use the memory efficient implementation? (default True)

        
        n_epochs (int) - number of epochs for training (default 300)
        batch_size (int) - size of minibatch (default 256)
        seed (int) - manually set the random seed (default None)
    """

    #path to save the results
    save = PFunctions.results_path


    # Get densenet configuration
    if (depth - 4) % 3:
        raise Exception('Invalid depth')
    block_config = [(depth - 4) // 6 for _ in range(3)]



    # create training data
    train_set = PFunctions.load_data_from_file(PFunctions.training_data_file_name)
    train_batch_list = np.array_split(train_set, batch_size_divider)

    # create test data
    test_set = PFunctions.load_data_from_file(PFunctions.test_data_file_name)
    test_batch_list = np.array_split(test_set, batch_size_divider)
  
    

    # Models
    model = DenseNet(
        growth_rate=growth_rate,
        block_config=block_config,
        num_init_features=growth_rate*2,
        num_classes=10,
        small_inputs=True,
        efficient=efficient,
    )
    print(model)
    
    # Print number of parameters
    num_params = sum(p.numel() for p in model.parameters())
    print("Total parameters: ", num_params)

    # Make save directory
    if not os.path.exists(save):
        os.makedirs(save)
    if not os.path.isdir(save):
        raise Exception('%s is not a dir' % save)

    # Train the model
    train(model=model, train_batch_list=train_batch_list, test_batch_list=test_batch_list, save=save,
          n_epochs=n_epochs, seed=seed)
    print('Done!')


"""
Try out the efficient DenseNet implementation:
python train_mode.py --efficient True  --save <path_to_save_dir>

Try out the naive DenseNet implementation:
python train_mode.py --efficient False  --save <path_to_save_dir>

Other args:
    --depth (int) - depth of the network (number of convolution layers) (default 40)
    --growth_rate (int) - number of features added per DenseNet layer (default 12)
    --n_epochs (int) - number of epochs for training (default 300)
    --batch_size (int) - size of minibatch (default 256)
    --seed (int) - manually set the random seed (default None)
"""
if __name__ == '__main__':
    fire.Fire(start_training)
