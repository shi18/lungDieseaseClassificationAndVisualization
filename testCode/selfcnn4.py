import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import argparse
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from plotConfusionMatrix import plot_confusion_matrix
import datetime
# import EarlyStopping
from pytorchtools import EarlyStopping
from torch.utils.data.sampler import SubsetRandomSampler

# Convolutional neural network (two convolutional layers)
class ConvNet(nn.Module):
    def __init__(self, num_classes=5):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.3)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.3)
        )
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.3)
        )

        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Sequential(
            nn.Linear(4608, 1024),
            nn.LeakyReLU(0.3),
            nn.Dropout2d(p=0.7)
        )

        # self.fc2 = nn.Sequential(
        #     nn.Linear(2048, 1024),
        #     nn.LeakyReLU(0.3),
        #     nn.Dropout2d(p=0.6)
        # )

        self.fc3 = nn.Sequential(
            nn.Linear(1024, 5),
            nn.Softmax()
        )

    def forward(self, x):
        # print('input shape:', x.shape)
        out = self.layer1(x)
        # print('after layer1: ', out.shape)
        out = self.layer2(out)
        # print('after layer1]2: ', out.shape)
        out = self.maxpool(out)
        # print('after maxpool: ', out.shape)
        out = self.layer3(out)
        # print('after layer3: ', out.shape)
        out = self.maxpool2(out)
        # print('after maxpool: ', out.shape)

        out = out.reshape(out.size(0), -1)
        # print('after reshape: ', out.size())
        out = self.fc1(out)
        # print('after fc1: ', out.size())
        out = self.fc3(out)
        # print('after fc2: ', out.size())
        return out


def create_datasets(args):
    batch_size = args.batch_size
    train_dir = args.train_dir
    test_dir = args.test_dir
    # percentage of training set to use as validation
    valid_size = 0.3

    # convert data to torch.FloatTensor
    transform = transforms.ToTensor()

    # choose the training and test datasets
    train_dataset = torchvision.datasets.ImageFolder(train_dir,
                                                     transform=torchvision.transforms.ToTensor())

    test_dataset = torchvision.datasets.ImageFolder(test_dir,
                                                    transform=torchvision.transforms.ToTensor())

    # obtain training indices that will be used for validation
    num_train = len(train_dataset)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(np.floor(valid_size * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]

    # define samplers for obtaining training and validation batches
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    # load training data in batches
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               sampler=train_sampler,
                                               num_workers=0)

    # load validation data in batches
    valid_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               sampler=valid_sampler,
                                               num_workers=0)

    # load test data in batches
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=batch_size,
                                              num_workers=0,
                                              shuffle=True)

    return train_loader, test_loader, valid_loader

def train_model(model, device, train_loader, valid_loader, test_loader, args):
    # Paras
    batch_size = args.batch_size  # 100; 2 is for fun
    learning_rate = args.lr
    num_epochs = args.epochs

    # # TRAINing
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=args.patience, verbose=True)
    # to track the training loss as the model trains
    train_losses = []
    # to track the validation loss as the model trains
    valid_losses = []
    # to track the average training loss per epoch as the model trains
    avg_train_losses = []
    # to track the average validation loss per epoch as the model trains
    avg_valid_losses = []

    # Train the model
    for epoch in range(num_epochs):
        perLoss = []
        y_true = []
        y_pred = []
        valid_true = []
        valid_pred = []
        model.train()
        for i, (images, labels) in enumerate(train_loader,1):
            images = images.to(device)
            labels = labels.to(device)

            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(images)
            # calculate the loss
            loss = criterion(output, labels)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # record training loss
            train_losses.append(loss.item())

            # log train acc
            _, predicted = torch.max(output.data, 1)

            y_pred_tmp = predicted.cpu().numpy().tolist()
            y_true_tmp = labels.cpu().numpy().tolist()
            y_true += y_true_tmp
            y_pred += y_pred_tmp


        ######################
        # validate the model #
        ######################
        model.eval()  # prep model for evaluation
        for data, target in valid_loader:
            data = data.to(device)
            target = target.to(device)
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the loss
            loss = criterion(output, target)
            # record validation loss
            valid_losses.append(loss.item())

            _, predicted = torch.max(output.data, 1)

            valid_pred_tmp = predicted.cpu().numpy().tolist()
            valid_true_tmp = target.cpu().numpy().tolist()
            valid_true += valid_true_tmp
            valid_pred += valid_pred_tmp
        trainAcc = accuracy_score(y_true, y_pred)
        validAcc = accuracy_score(valid_true, valid_pred)

        # print training/validation statistics
        # calculate avAerage loss over an epoch
        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)



        # print_msg = (f'[{epoch:>{num_epochs}}/{n_epochs:>{num_epochs}}]' +
        #              f'train_loss: {train_loss:.5f} ' +
        #              f'valid_loss: {valid_loss:.5f}')
        print('epoch:', epoch, 'trainLoss: ',round(train_loss, 4),
              ' validationLoss: ', round(valid_loss, 4),
              'trainAcc:', round(trainAcc,2),
              'validationAcc', round(validAcc, 2))

        # clear lists to track next epoch
        train_losses = []
        valid_losses = []

        # early_stopping needs the validation loss to check if it has decresed,
        # and if it has, it will make a checkpoint of the current model
        early_stopping(valid_loss, model)

        if early_stopping.early_stop:
            print("Early stopping")
            break
    # load the last checkpoint with the best model
    model.load_state_dict(torch.load('checkpoint.pt'))

    return model, avg_train_losses, avg_valid_losses

def test_model(model, device, test_loader, args):
    # Paras
    test_dir = args.test_dir
    batch_size = args.batch_size
    y_true = []
    y_pred = []

    # Test the model
    model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            y_pred_tmp = predicted.cpu().numpy().tolist()
            y_true_tmp = labels.cpu().numpy().tolist()
            y_true += y_true_tmp
            y_pred += y_pred_tmp

    # Save the model checkpoint
    currentDT = str(datetime.datetime.now())[0:19]
    modelName = currentDT + 'model.pt'
    print('model name: ', modelName)
    torch.save(model.state_dict(), modelName)
    return y_true, y_pred


def main():
    parser = argparse.ArgumentParser(description='CNN Example')
    parser.add_argument('train_dir', help='Root dir for training data.')
    parser.add_argument('test_dir', help='Root dir for testing data.')
    parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--num_classes', type=int, default=5,
                        help='Number of classes.')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate.')
    parser.add_argument('--augment', action='store_true', default=False,
                        help='Use augment images for training')
    parser.add_argument('--aug_dir', help='Root dir for extra training data.')
    parser.add_argument('--class_name', default=['healthy', 'emphysema', 'ground_glass', 'fibrosis', 'micronodules'],
                        help='Use classes name for display')
    parser.add_argument('--patience', type=int, default=5,
                        help='set patience to control early stoppinp')

    args = parser.parse_args()

    # Hyper parameters
    img_train = args.train_dir
    num_classes = args.num_classes
    # Device configuration
    device = torch.device('cuda:0')# if torch.cuda.is_available() else 'cpu')
    model = ConvNet(num_classes).to(device)
    train_loader, test_loader, valid_loader = create_datasets(args)
    model, train_loss, valid_loss = train_model(model, device, train_loader,
                                                valid_loader, test_loader, args)    # save epochsLoss as npy file

    # if args.augment:
    #     aug_dir = args.aug_dir
    #     train_model(model, device, aug_dir, args)

    y_true, y_pred = test_model(model, device, test_loader, args)
    print(accuracy_score(y_true, y_pred))
    target_names = args.class_name
    print(classification_report(y_true, y_pred, target_names=target_names))

    plot_confusion_matrix(y_true, y_pred,
                          classes=['healthy', 'emphysema', 'ground_glass', 'fibrosis', 'micronodules'],
                          title='confusion matrix')
    plt.show()


if __name__ == '__main__':
    main()
