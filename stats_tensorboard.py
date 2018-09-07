import torch

LIST_ACTIVATIONS_TENSORBOARD = ['conv1', 'primary_caps', 'digit_caps']

def compute_save_stats(writer, train_loader, capsnet, criterion_capsnet, optimizer):
    #tensorboard: compute stats on first batch (taken randomly among all training images)
    inputs, labels = next(iter(train_loader))
    inputs = inputs.to(device)
    labels = labels.to(device)
    capsnet.eval()
    for name, param in capsnet.named_parameters():
        writer.add_histogram('weights/'+name, param.clone().cpu().data.numpy(), epoch)
    temp = inputs.clone()
    for name in LIST_ACTIVATIONS_TENSORBOARD:
        temp = capsnet._modules[name](temp)
        writer.add_histogram('activation/'+name, temp.cpu().data.numpy(), epoch)
    del temp
    torch.cuda.empty_cache()
    output_norm, _ = capsnet(inputs)
    writer.add_histogram('activation/outputs', output_norm.cpu().data.numpy(), epoch)
    loss = criterion_capsnet(output_norm, labels)
    optimizer.zero_grad()
    loss.backward()
    for name, param in capsnet.named_parameters():
        writer.add_histogram('grad/'+name, param.grad.clone().cpu().data.numpy(), epoch)