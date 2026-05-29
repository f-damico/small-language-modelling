import torch

def evaluate( model, criterion, dataset, iters, device):
    """
    Evaluate criterion by sampling iters batches from dataset.
    
    Returns: TODO explain text data and criterion

    """
    dataset.reset()
    model.eval()
    running = 0.
    
    with torch.no_grad():
        for _ in range(iters):

            inputs, targets = dataset.next_batch()
            inputs, targets = inputs.to(device), targets.to(device)

#             with torch.autocast(device_type=device, dtype=torch.bfloat16): #TODO: check that this is actually running faster on gpus
#                 logits = model(inputs)
#                 loss = criterion(logits.view(-1, logits.size(-1)),targets.view(-1))
            logits = model(inputs)
            loss = criterion(logits.view(-1, logits.size(-1)),targets.view(-1))

            running += loss.item()

    return running/iters

def loss_by_token( model, criterion, dataset, block_size, iters, device):
    """
    Evaluate criterion by sampling iters batches from dataset.
    
    Returns: TODO explain text data and criterion

    """
    dataset.reset()
    model.eval()
    running = torch.zeros(block_size)
    
    with torch.no_grad():
        for _ in range(iters):

            inputs, targets = dataset.next_batch()
            inputs, targets = inputs.to(device), targets.to(device)

#             with torch.autocast(device_type=device, dtype=torch.bfloat16): #TODO: check that this is actually running faster on gpus
#                 logits = model(inputs)
#                 loss = criterion(logits.view(-1, logits.size(-1)),targets.view(-1))
            logits = model(inputs)

            losses = torch.zeros(block_size)
            for i in range(block_size):
                losses[i] = criterion(logits[:,i,:], targets[:,i])
            running += losses

    return running/iters