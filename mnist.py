import torch, torchvision, os, sys
from capsnet import CapsNet

DATA_FOLDER = 'data/'
BATCH_SIZE = 500
NB_EPOCHS = 100

nb_cores = os.cpu_count()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])
MNIST_data = torchvision.datasets.MNIST(root=DATA_FOLDER, train=True, transform=transform, target_transform=None, download=True)
data_loader = torch.utils.data.DataLoader(MNIST_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=nb_cores)
model = CapsNet().to(device)
criterion = torch.nn.NLLLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

print('Number of parameters:', model.count_parameters())
for epoch in range(NB_EPOCHS):
    print('Epoch {}/{}'.format(epoch + 1, NB_EPOCHS))

    model.train()
    running_loss = 0.0
    running_corrects = 0
    nb_images = 0

    for i, (inputs, labels) in enumerate(data_loader):
        # print(i)
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.shape[0]
        running_corrects += torch.sum(preds == labels.data)
        nb_images += inputs.shape[0]

    epoch_loss = running_loss / nb_images
    epoch_acc = running_corrects.double() / nb_images

    print('Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))
    print('-' * 100)

