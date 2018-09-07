import torch

# LIST_ACTIVATIONS_TENSORBOARD = ['conv1', 'primary_caps', 'digit_caps']
LIST_ACTIVATIONS_TENSORBOARD = ['capsnet']

def compute_save_stats(writer, train_loader, model, criterion, optimizer, device, epoch):
    #tensorboard: compute stats on first batch (taken randomly among all training images)
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