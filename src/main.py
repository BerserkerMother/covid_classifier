from argparse import ArgumentParser
import logging

import torch
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader, random_split

from data import CovidDataset
from model import CovidModel
from utils import AverageMeter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args):
    # image transforms
    data_transforms = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])
    # creating dataset
    dataset = CovidDataset(root=args.data, transform=data_transforms)
    # splitting the dataset
    num_data = len(dataset)
    train_size = int(num_data * 0.7)
    val_size = int(num_data * 0.15)
    test_size = num_data - (train_size + val_size)
    train_set, val_set, test_set = \
        random_split(dataset, lengths=[train_size, val_size, test_size])
    # data loaders
    train_loader = DataLoader(
        dataset=train_set,
        shuffle=True,
        batch_size=args.batch_size
    )
    val_loader = DataLoader(
        dataset=val_set,
        shuffle=True,
        batch_size=args.batch_size
    )
    test_loader = DataLoader(
        dataset=test_set,
        shuffle=True,
        batch_size=args.batch_size
    )

    # define the model
    model = CovidModel(pretrained=True).to(device)
    # define optimizer
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr, weight_decay=args.weight_decay)

    # saving the best model
    best_acc = 0.
    # training loop
    for e in range(1, args.epochs + 1):
        train_loss, train_acc = train(train_loader, model, optimizer)
        val_loss, val_acc = val(val_loader, model)

        logging.info("Epoch [%3d/%3d], train loss: %4.4f, val loss: %4.4f,"
                     " train acc: %2.2f%%, val acc: %2.2f%%" %
                     (e, args.epochs, train_loss, val_loss, train_acc, val_acc))
        # saving model if val acc is higher than the best so far
        if val_loss > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "model.pth")

    # now load the best model and evaluate test set
    _, test_acc = val(test_loader, model)
    logging.info("test accuracy %2.2f%%" % test_acc)


def train(loader, model, optimizer):
    """

    :param loader: data loader
    :param model: basically the model
    :param optimizer: basically optimizer
    :return: train loss and train accuracy
    """
    model.train()  # enable train mode
    train_acc = AverageMeter()
    train_loss = AverageMeter()
    temp_loss = 0.
    for i, (images, labels) in enumerate(loader):
        batch_size = images.size()[0]
        # move data to device
        images = images.to(device)
        labels = labels.to(device).type(torch.float)

        # forward pass and calc loss
        logits = model(images)
        loss = F.binary_cross_entropy(logits, labels)

        pred = torch.where(logits >= 0.5, 1., 0.)
        num_correct = (pred == labels).sum()
        train_acc.update(num_correct, num=batch_size)
        train_loss.update(loss.item())
        temp_loss += loss.item()

        # optimizing step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 3 == 0:
            logging.info("step %d, loss: %4.4f" % (i + 1, temp_loss / 3))
            temp_loss = 0.
    return train_loss.avg(), train_acc.avg() * 100


def val(loader, model):
    """

    :param loader: basic the data loader
    :param model: basically the model
    :return: loss and accuracy
    """

    val_acc = AverageMeter()
    val_loss = AverageMeter()
    for images, labels in loader:
        batch_size = images.size()[0]
        # move data to device
        images = images.to(device)
        labels = labels.to(device).type(torch.float)

        # forward pass and calc loss
        logits = model(images)
        loss = F.binary_cross_entropy(logits, labels)

        pred = torch.where(logits >= 0.5, 1., 0.)
        num_correct = (pred == labels).sum()
        val_acc.update(num_correct, num=batch_size)
        val_loss.update(loss.item())
    return val_loss.avg(), val_acc.avg() * 100


arg_parser = ArgumentParser(description=
                            "Covid classifier based onCT scan image")
# data
arg_parser.add_argument("--data", type=str, default="data",
                        help="path to dataset main folder")
arg_parser.add_argument("--batch_size", type=int, default=32,
                        help="size of batch")
# model
arg_parser.add_argument("--lr", type=float, default=1e-4,
                        help="optimizer learning rate")
arg_parser.add_argument("--weight_decay", type=float, default=1e-5,
                        help="optimizer weight_decay")
# other
arg_parser.add_argument("--epochs", type=int, default=25,
                        help="number of training epochs")
arguments = arg_parser.parse_args()
logging.basicConfig(
    format='[%(levelname)s] - %(message)s',
    level=logging.INFO
)

if __name__ == "__main__":
    main(arguments)
