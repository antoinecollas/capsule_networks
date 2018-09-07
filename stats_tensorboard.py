import torch, random
from torchvision.utils import make_grid

# LIST_ACTIVATIONS_TENSORBOARD = ['conv1', 'primary_caps', 'digit_caps']
LIST_ACTIVATIONS_TENSORBOARD = ['capsnet']
NB_IM_TO_SAVE = 16 #must be smaller than BATCH_SIZE

def begin_epoch_stats(writer, train_loader, val_loader, model, criterion, optimizer, device, epoch):
    #tensorboard: compute stats on first batch of training set (taken randomly among all training images)
    inputs, labels = next(iter(train_loader))
    inputs = inputs.to(device)
    labels = labels.to(device)
    model.eval()
    for name, param in model.named_parameters():
        writer.add_histogram('weights/'+name, param.clone().cpu().data.numpy(), epoch)
    temp = inputs.clone()
    for name in LIST_ACTIVATIONS_TENSORBOARD:
        temp = model._modules[name](temp)
        if type(temp) is tuple:
            temp = temp[0]
        writer.add_histogram('activation/'+name, temp.cpu().data.numpy(), epoch)
    output_norm, reconstructed_image = model(inputs)
    writer.add_histogram('activation/outputs', output_norm.cpu().data.numpy(), epoch)
    loss = criterion(inputs, reconstructed_image, labels, output_norm)
    optimizer.zero_grad()
    loss[0].backward()
    for name, param in model.named_parameters():
        writer.add_histogram('grad/'+name, param.grad.clone().cpu().data.numpy(), epoch)

    #tensorboard: save images on first batch of validation set (taken randomly among all validation images)
    #(originals and reconstructed)
    inputs, _ = next(iter(val_loader))
    inputs = inputs.to(device)
    model.eval()
    im_to_save = random.sample(range(inputs.shape[0]), NB_IM_TO_SAVE)
    inputs = inputs.reshape((inputs.shape[0], 1, inputs.shape[-2], inputs.shape[-1]))[im_to_save]
    im = make_grid(inputs, normalize=False, scale_each=False)
    writer.add_image('val/orgininal', im, epoch)
    reconstructed_image = reconstructed_image.reshape((reconstructed_image.shape[0], 1, inputs.shape[-2], inputs.shape[-1]))[im_to_save]
    im = make_grid(reconstructed_image, normalize=False, scale_each=False)
    writer.add_image('val/reconstructed', im, epoch)