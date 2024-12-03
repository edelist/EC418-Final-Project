# from planner import Planner, save_model 
# import torch
# import torch.utils.tensorboard as tb
# import numpy as np
# from utils import load_data
# import dense_transforms

# def train(args):
#     from os import path
#     model = Planner()
#     train_logger, valid_logger = None, None
#     if args.log_dir is not None:
#         train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'))

#     """
#     Your code here, modify your HW4 code
    
#     """
#     import torch

#     device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
	
#     print("device:", device)
#     model = model.to(device)
#     if args.continue_training:
#         model.load_state_dict(torch.load(path.join(path.dirname(path.abspath(__file__)), 'planner.th')))

#     loss = torch.nn.L1Loss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    
#     import inspect
#     transform = eval(args.transform, {k: v for k, v in inspect.getmembers(dense_transforms) if inspect.isclass(v)})

#     train_data = load_data('drive_data', transform=transform, num_workers=args.num_workers)

#     global_step = 0
#     for epoch in range(args.num_epoch):
#         model.train()
#         losses = []
#         for img, label in train_data:
#             img, label = img.to(device), label.to(device)

#             pred = model(img)
#             loss_val = loss(pred, label)

#             if train_logger is not None:
#                 train_logger.add_scalar('loss', loss_val, global_step)
#                 if global_step % 100 == 0:
#                     log(train_logger, img, label, pred, global_step)

#             optimizer.zero_grad()
#             loss_val.backward()
#             optimizer.step()
#             global_step += 1
            
#             losses.append(loss_val.detach().cpu().numpy())
        
#         avg_loss = np.mean(losses)
#         if train_logger is None:
#             print('epoch %-3d \t loss = %0.3f' % (epoch, avg_loss))
#         save_model(model)

#     save_model(model)

# def log(logger, img, label, pred, global_step):
#     """
#     logger: train_logger/valid_logger
#     img: image tensor from data loader
#     label: ground-truth aim point
#     pred: predited aim point
#     global_step: iteration
#     """
#     import matplotlib.pyplot as plt
#     import torchvision.transforms.functional as TF
#     fig, ax = plt.subplots(1, 1)
#     ax.imshow(TF.to_pil_image(img[0].cpu()))
#     WH2 = np.array([img.size(-1), img.size(-2)])/2
#     ax.add_artist(plt.Circle(WH2*(label[0].cpu().detach().numpy()+1), 2, ec='g', fill=False, lw=1.5))
#     ax.add_artist(plt.Circle(WH2*(pred[0].cpu().detach().numpy()+1), 2, ec='r', fill=False, lw=1.5))
#     logger.add_figure('viz', fig, global_step)
#     del ax, fig



# if __name__ == '__main__':
#     import argparse

#     parser = argparse.ArgumentParser()

#     parser.add_argument('--log_dir')
#     # Put custom arguments here
#     parser.add_argument('-n', '--num_epoch', type=int, default=30)
#     parser.add_argument('-w', '--num_workers', type=int, default=4)
#     parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3)
#     parser.add_argument('-c', '--continue_training', action='store_true')
#     parser.add_argument('-t', '--transform', default='Compose([ColorJitter(0.2, 0.5, 0.5, 0.2), RandomHorizontalFlip(), ToTensor()])')

#     args = parser.parse_args()
#     train(args)

from planner import Planner, save_model
import torch
import torch.utils.tensorboard as tb
import numpy as np
from utils import load_data
import dense_transforms
from torch.utils.data import random_split, DataLoader


def train(args):
    from os import path

    model = Planner()
    train_logger = None
    val_logger = None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'))
        val_logger = tb.SummaryWriter(path.join(args.log_dir, 'val'))

    # Setting the device
    device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = model.to(device)
    if args.continue_training:
        model.load_state_dict(torch.load(path.join(path.dirname(path.abspath(__file__)), 'planner.th')))

    # Loss and optimizer
    loss_fn = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # Scheduler for learning rate decay
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    # Data augmentation and loading
    import inspect
    transform = eval(args.transform, {k: v for k, v in inspect.getmembers(dense_transforms) if inspect.isclass(v)})

    # Ensure `load_data` returns a Dataset and split manually
    dataset = load_data('drive_data', transform=transform)  # Assuming `load_data` now returns a Dataset
    if isinstance(dataset, DataLoader):
        dataset = dataset.dataset  # Extract Dataset if `load_data` returns DataLoader

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    global_step = 0
    best_val_loss = float('inf')
    early_stop_counter = 0

    for epoch in range(args.num_epoch):
        # Training
        model.train()
        train_losses = []
        for img, label in train_loader:
            img, label = img.to(device), label.to(device)

            # Forward pass
            pred = model(img)
            loss_val = loss_fn(pred, label)

            # Backward pass
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()

            train_losses.append(loss_val.item())
            if train_logger:
                train_logger.add_scalar('training_loss', loss_val.item(), global_step)

            global_step += 1

        train_loss = np.mean(train_losses)

        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for img, label in val_loader:
                img, label = img.to(device), label.to(device)
                pred = model(img)
                loss_val = loss_fn(pred, label)
                val_losses.append(loss_val.item())

        val_loss = np.mean(val_losses)

        # Update the learning rate
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        # Log metrics
        if train_logger:
            train_logger.add_scalar('epoch_train_loss', train_loss, epoch)
            train_logger.add_scalar('epoch_val_loss', val_loss, epoch)
            train_logger.add_scalar('learning_rate', current_lr, epoch)

        if val_logger:
            val_logger.add_scalar('val_loss', val_loss, epoch)

        # Print progress
        print(f"Epoch {epoch+1}/{args.num_epoch} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {current_lr:.6f}")

        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_model(model)
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= args.early_stop_patience:
                print("Early stopping triggered.")
                break


def log(logger, img, label, pred, global_step):
    """
    logger: train_logger/val_logger
    img: image tensor from data loader
    label: ground-truth aim point
    pred: predicted aim point
    global_step: iteration
    """
    import matplotlib.pyplot as plt
    import torchvision.transforms.functional as TF

    fig, ax = plt.subplots(1, 1)
    ax.imshow(TF.to_pil_image(img[0].cpu()))
    WH2 = np.array([img.size(-1), img.size(-2)]) / 2
    ax.add_artist(plt.Circle(WH2 * (label[0].cpu().detach().numpy() + 1), 2, ec='g', fill=False, lw=1.5))
    ax.add_artist(plt.Circle(WH2 * (pred[0].cpu().detach().numpy() + 1), 2, ec='r', fill=False, lw=1.5))
    logger.add_figure('viz', fig, global_step)
    del ax, fig


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir', help="Directory to save logs")
    parser.add_argument('-n', '--num_epoch', type=int, default=30, help="Number of training epochs")
    parser.add_argument('-b', '--batch_size', type=int, default=32, help="Batch size for training and validation")
    parser.add_argument('-w', '--num_workers', type=int, default=4, help="Number of worker threads for data loading")
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3, help="Learning rate")
    parser.add_argument('-c', '--continue_training', action='store_true', help="Continue training from last checkpoint")
    parser.add_argument('-t', '--transform', default='Compose([ColorJitter(0.2, 0.5, 0.5, 0.2), RandomHorizontalFlip(), ToTensor()])', help="Data augmentation pipeline")
    parser.add_argument('--early_stop_patience', type=int, default=10, help="Early stopping patience")

    args = parser.parse_args()
    train(args)
