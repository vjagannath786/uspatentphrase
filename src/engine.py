from tqdm import tqdm
import torch
import config


def train_fn(model, data_loader, optimizer, scheduler):
    model.train()
    fin_loss = 0
    accum_iter = 1
    tk0 = tqdm(data_loader, total=len(data_loader))
    
    for data in tk0:

        for key, value in data.items():
            data[key] = value.to(config.device)
        
        
        
        _, loss, metrics = model(**data)
        
        
        optimizer.zero_grad()
        loss.backward()

        # weights update
        #if ((batch_idx + 1) % accum_iter == 0) or (batch_idx + 1 == len(data_loader)):
        #    #print('in accumulation')
            
            
            
        optimizer.step()
        scheduler.step()
        #model.zero_grad()
            
        
        
        fin_loss += loss.item()
        
        
    return fin_loss / len(data_loader), metrics


def eval_fn(model, data_loader):
    model.eval()
    fin_loss = 0
    fin_preds = []
    tk0 = tqdm(data_loader, total=len(data_loader))
    with torch.no_grad():
        for data in tk0:
            #a
            for key, value in data.items():
                #a
                data[key] = value.to(config.device)
            batch_preds, loss, metrics = model(**data)
            fin_loss += loss.item()
            fin_preds.append(batch_preds.cpu().detach().numpy())

    
    return fin_preds, fin_loss / len(data_loader), metrics