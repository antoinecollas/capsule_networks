import torch, torchvision, os, sys, random, math
from capsnet import CapsNet, MarginLoss
from torch.utils.data import DataLoader, SubsetRandomSampler
from tensorboardX import SummaryWriter

DATA_FOLDER = 'data/'
BATCH_SIZE = 500
NB_EPOCHS = 1000
SIZE_TRAIN_SET = 60000
PROPORTION_TRAIN_SET = 0.8

writer = SummaryWriter()

nb_cores = os.cpu_count()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])
MNIST_data = torchvision.datasets.MNIST(root=DATA_FOLDER, train=True, transform=transform, target_transform=None, download=True)
train_indices = set(random.sample(range(SIZE_TRAIN_SET), math.floor((SIZE_TRAIN_SET+1)*PROPORTION_TRAIN_SET)))
val_indices = set(range(SIZE_TRAIN_SET)) - train_indices
train_indices = list(train_indices)
val_indices = list(val_indices)
train_sampler = SubsetRandomSampler(train_indices)
val_sampler = SubsetRandomSampler(val_indices)
train_loader = DataLoader(MNIST_data, batch_size=BATCH_SIZE, sampler=train_sampler, num_workers=nb_cores)
val_loader = DataLoader(MNIST_data, batch_size=BATCH_SIZE, sampler=val_sampler, num_workers=nb_cores)

model = CapsNet().to(device)
# criterion = torch.nn.NLLLoss()
criterion = MarginLoss(0.9, 0.1, 0.5)
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

print('Number of parameters:', model.count_parameters())
for epoch in range(NB_EPOCHS):
    print('Epoch {}/{}'.format(epoch + 1, NB_EPOCHS))

    running_loss = 0.0
    running_corrects = 0
    nb_images = 0
    model.train()
    for i, (inputs, labels) in enumerate(train_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        # for param in model.parameters():
        #     print(param.grad)
        optimizer.step()
        running_loss += loss.item() * inputs.shape[0]
        running_corrects += torch.sum(preds == labels.data)
        nb_images += inputs.shape[0]
        batch_acc = torch.sum(preds == labels.data).double()/inputs.shape[0]
        # print('Train set: batch n {}. Loss: {:.3f} Acc: {:.3f}'.format(i, loss.item(), batch_acc))

    epoch_loss = running_loss / nb_images
    epoch_acc = running_corrects.double() / nb_images

    running_loss = 0.0
    running_corrects = 0
    nb_images = 0
    model.eval()
    for i, (inputs, labels) in enumerate(val_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)
        running_loss += loss.item() * inputs.shape[0]
        running_corrects += torch.sum(preds == labels.data)
        nb_images += inputs.shape[0]
        batch_acc = torch.sum(preds == labels.data).double()/inputs.shape[0]
        # print('Val set: batch n {}. Loss: {:.3f} Acc: {:.3f}'.format(i, loss.item(), batch_acc))

    epoch_loss_val = running_loss / nb_images
    epoch_acc_val = running_corrects.double() / nb_images
    writer.add_scalar('train/Loss', epoch_loss, epoch)
    writer.add_scalar('train/Acc', epoch_acc, epoch)
    writer.add_scalar('val/Loss', epoch_loss_val, epoch)
    writer.add_scalar('val/Acc', epoch_acc_val, epoch)
    print('Total: Train set: Loss: {:.4f} Acc: {:.4f} Val set: Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc, epoch_loss_val, epoch_acc_val))
    print('-' * 20)

writer.close()