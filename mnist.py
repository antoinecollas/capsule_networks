import torch, torchvision, os, sys, random, math
from torch.nn import MSELoss
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import transforms
from tensorboardX import SummaryWriter
from capsnet import Model, TotalLoss
from stats_tensorboard import begin_epoch_stats

DATA_FOLDER = 'data/'
BATCH_SIZE = 200
NB_EPOCHS = 1000
SIZE_TRAIN_SET = 60000
PROPORTION_TRAIN_SET = 0.8
writer = SummaryWriter()

nb_cores = os.cpu_count()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

training_transform = transforms.Compose([
    transforms.RandomAffine(degrees=0, translate=(0.1,0.1), scale=None, shear=None, resample=False, fillcolor=0),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
validation_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
training_MNIST_data = torchvision.datasets.MNIST(root=DATA_FOLDER, train=True, transform=training_transform, target_transform=None, download=True)
validation_MNIST_data = torchvision.datasets.MNIST(root=DATA_FOLDER, train=True, transform=validation_transform, target_transform=None, download=True)
size_train_set = math.floor((SIZE_TRAIN_SET+1)*PROPORTION_TRAIN_SET)
train_indices = set(random.sample(range(SIZE_TRAIN_SET), size_train_set))
val_indices = set(range(SIZE_TRAIN_SET)) - train_indices
train_indices = list(train_indices)
print('len(train_indices)=', len(train_indices))
val_indices = list(val_indices)
print('len(val_indices)=', len(val_indices))
train_sampler = SubsetRandomSampler(train_indices)
val_sampler = SubsetRandomSampler(val_indices)
train_loader = DataLoader(training_MNIST_data, batch_size=BATCH_SIZE, sampler=train_sampler, num_workers=nb_cores)
val_loader = DataLoader(validation_MNIST_data, batch_size=BATCH_SIZE, sampler=val_sampler, num_workers=nb_cores)

model = Model().to(device)
print('Number of parameters:', model.count_parameters())

criterion = TotalLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

for epoch in range(NB_EPOCHS):
    print('Epoch {}/{}'.format(epoch + 1, NB_EPOCHS))

    begin_epoch_stats(writer, train_loader, val_loader, model, criterion, optimizer, device, epoch)
    
    #training
    running_loss = 0.0
    running_corrects = 0
    nb_images = 0
    model.train()
    for i, (inputs, labels) in enumerate(train_loader):
        #forward
        inputs = inputs.to(device)
        labels = labels.to(device)
        output_norm, reconstructed_image = model(inputs, labels)
        _, preds = torch.max(output_norm, 1)
        loss, margin_loss, mse = criterion(inputs, reconstructed_image, labels, output_norm)
        
        #backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #stats
        running_loss += loss.item() * inputs.shape[0]
        running_corrects += torch.sum(preds == labels.data)
        nb_images += inputs.shape[0]
        batch_acc = torch.sum(preds == labels.data).double()/inputs.shape[0]
        print('Train set: batch n {}. Loss: {:.3f} Acc: {:.3f}'.format(i, loss.item(), batch_acc))

    epoch_loss = running_loss / nb_images
    epoch_acc = running_corrects.double() / nb_images

    #validation
    running_loss = 0.0
    running_corrects = 0
    nb_images = 0
    model.eval()
    for i, (inputs, labels) in enumerate(val_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        output_norm, reconstructed_image = model(inputs)
        _, preds = torch.max(output_norm, 1)
        loss, margin_loss, mse = criterion(inputs, reconstructed_image, labels, output_norm)
        running_loss += loss.item() * inputs.shape[0]
        running_corrects += torch.sum(preds == labels.data)
        nb_images += inputs.shape[0]
        batch_acc = torch.sum(preds == labels.data).double()/inputs.shape[0]
        # print('Val set: batch n {}. Loss: {:.3f} Acc: {:.3f}'.format(i, loss.item(), batch_acc))

    #tensorboard
    epoch_loss_val = running_loss / nb_images
    epoch_acc_val = running_corrects.double() / nb_images
    writer.add_scalar('train/Loss', epoch_loss, epoch)
    writer.add_scalar('train/Acc', epoch_acc, epoch)
    writer.add_scalar('val/Loss', epoch_loss_val, epoch)
    writer.add_scalar('val/Acc', epoch_acc_val, epoch)
    print('Total: Train set: Loss: {:.4f} Acc: {:.4f} Val set: Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc, epoch_loss_val, epoch_acc_val))
    print('-' * 20)

writer.close()