# from planner import Planner, save_model 
# import torch
# import torch.utils.tensorboard as tb
# import numpy as np
# from utils import load_data
# import dense_transforms
# import matplotlib.pyplot as plt

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
#     loss_history = []
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
#         loss_history.append(avg_loss)
#         if train_logger is None:
#             print('epoch %-3d \t loss = %0.3f' % (epoch, avg_loss))
#         save_model(model)

#     # Plot the loss curve after training
#     plt.figure()
#     plt.plot(range(len(loss_history)), loss_history, label='Training Loss')
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     plt.title('Training Loss Curve')
#     plt.legend()
#     plt.grid(True)
#     plt.show()

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

# from planner import PlannerCBAM, save_model
# import torch
# import torch.utils.tensorboard as tb
# import numpy as np
# from utils import load_data
# import dense_transforms
# import matplotlib.pyplot as plt
# from torchvision.transforms import Compose, ColorJitter, RandomHorizontalFlip, ToTensor


# def train(args):
#     from os import path
#     model = PlannerCBAM()
#     train_logger, valid_logger = None, None

#     if args.log_dir is not None:
#         train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'))

#     device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
#     print("device:", device)

#     model = model.to(device)
#     if args.continue_training:
#         model.load_state_dict(torch.load(path.join(path.dirname(path.abspath(__file__)), 'planner_cbam.th')))

#     loss_fn = torch.nn.L1Loss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

#     transform = eval(args.transform, globals())

#     train_loader = load_data('drive_data', transform=transform, num_workers=args.num_workers)

#     global_step = 0
#     for epoch in range(args.num_epoch):
#         model.train()
#         losses = []

#         for img, label in train_loader:
#             img, label = img.to(device), label.to(device)

#             pred = model(img)
#             loss_val = loss_fn(pred, label)

#             optimizer.zero_grad()
#             loss_val.backward()
#             optimizer.step()

#             losses.append(loss_val.detach().cpu().numpy())
#             global_step += 1

#             if train_logger is not None:
#                 train_logger.add_scalar('loss', loss_val, global_step)

#         avg_loss = np.mean(losses)
#         print(f'Epoch {epoch + 1}/{args.num_epoch} | Loss: {avg_loss:.4f}')

#     save_model(model)
#     plot_loss_curve(losses)


# def plot_loss_curve(losses):
#     plt.figure()
#     plt.plot(losses, label="Training Loss")
#     plt.xlabel("Iteration")
#     plt.ylabel("Loss")
#     plt.title("Loss Curve")
#     plt.legend()
#     plt.show()


# if __name__ == '__main__':
#     import argparse

#     parser = argparse.ArgumentParser()

#     parser.add_argument('--log_dir')
#     parser.add_argument('-n', '--num_epoch', type=int, default=30)
#     parser.add_argument('-w', '--num_workers', type=int, default=4)
#     parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3)
#     parser.add_argument('-c', '--continue_training', action='store_true')
#     parser.add_argument('-t', '--transform', default='Compose([ColorJitter(0.2, 0.5, 0.5, 0.2), RandomHorizontalFlip(), ToTensor()])')

#     args = parser.parse_args()
#     train(args)

from planner import PlannerCBAM, save_model
import torch
import torch.utils.tensorboard as tb
import numpy as np
from utils import load_data, SuperTuxDataset
import dense_transforms
import matplotlib.pyplot as plt
from torch.utils.data import random_split, DataLoader
from torchvision.transforms import Compose, ColorJitter, RandomHorizontalFlip, ToTensor


def train(args):
    from os import path
    model = PlannerCBAM()
    train_logger, valid_logger = None, None

    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'))

    device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
    print("device:", device)

    model = model.to(device)
    if args.continue_training:
        model.load_state_dict(torch.load(path.join(path.dirname(path.abspath(__file__)), 'planner_cbam.th')))

    loss_fn = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # Dataset with train/test split
    transform = eval(args.transform, globals())
    dataset = SuperTuxDataset(dataset_path='drive_data', transform=transform)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_data, test_data = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_data, batch_size=128, shuffle=True, num_workers=args.num_workers)
    test_loader = DataLoader(test_data, batch_size=128, shuffle=False, num_workers=args.num_workers)

    train_losses, test_losses = [], []

    for epoch in range(args.num_epoch):
        # Training phase
        model.train()
        train_loss = 0
        for img, label in train_loader:
            img, label = img.to(device), label.to(device)

            pred = model(img)
            loss_val = loss_fn(pred, label)

            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()

            train_loss += loss_val.item()

            if train_logger is not None:
                train_logger.add_scalar('train_loss', loss_val, epoch)

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # Validation phase
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for img, label in test_loader:
                img, label = img.to(device), label.to(device)

                pred = model(img)
                loss_val = loss_fn(pred, label)
                test_loss += loss_val.item()

        test_loss /= len(test_loader)
        test_losses.append(test_loss)

        print(f"Epoch {epoch + 1}/{args.num_epoch} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}")

    save_model(model)
    plot_loss_curve(train_losses, test_losses)


def plot_loss_curve(train_losses, test_losses):
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(test_losses, label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train vs Test Loss Curve")
    plt.legend()
    plt.savefig('loss_curve.png')
    plt.show()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    parser.add_argument('-n', '--num_epoch', type=int, default=30)
    parser.add_argument('-w', '--num_workers', type=int, default=4)
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3)
    parser.add_argument('-c', '--continue_training', action='store_true')
    parser.add_argument('-t', '--transform', default='Compose([ColorJitter(0.2, 0.5, 0.5, 0.2), RandomHorizontalFlip(), ToTensor()])')

    args = parser.parse_args()
    train(args)
