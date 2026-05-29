import torch

def train_step( model, trainset, criterion, optimizer, scheduler, device):
    """
    One step of gradient descent.
    
    Returns:
        Current loss.
    """
    model.train()
    optimizer.zero_grad()

    inputs, targets = trainset.next_batch()
    inputs, targets = inputs.to(device), targets.to(device)

#     with torch.autocast(device_type=device, dtype=torch.bfloat16): #TODO: check that this is actually running faster on gpus
#         logits = model(inputs)
#         loss = criterion(logits.view(-1, logits.size(-1)),targets.view(-1))
    logits = model(inputs)
    loss = criterion(logits.view(-1, logits.size(-1)),targets.view(-1))

    loss.backward()
    optimizer.step()
    scheduler.step()

    return loss.item()
