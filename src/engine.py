from tqdm import tqdm
import torch
import config


def train_fn(model, data_loader, optimizer, scheduler):
    model.train()
    fin_loss = 0
    accum_iter = 1
    tk0 = tqdm(data_loader, total=len(data_loader))
    
    for i, (input_ids, attention_mask, targets) in enumerate(tk0):

        #for key, value in data.items():
        #    data[key] = value.to(config.device)
        
        input_ids = input_ids.to(config.device)
        attention_mask = attention_mask.to(config.device)
        targets = targets.to(config.device)
        
        _, loss, metrics = model(input_ids, attention_mask, targets)
        
        
        
        loss.backward()

        # weights update
        #if ((batch_idx + 1) % accum_iter == 0) or (batch_idx + 1 == len(data_loader)):
        #    #print('in accumulation')
            
            
            
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
            
        
        
        fin_loss += loss.item()
        
        
    return fin_loss / len(data_loader), metrics


def eval_fn(model, data_loader):
    model.eval()
    fin_loss = 0
    fin_preds = []
    tk0 = tqdm(data_loader, total=len(data_loader))
    with torch.no_grad():
        for i, (input_ids, attention_mask, targets) in enumerate(tk0):
            #a
            #for key, value in data.items():
                #a
            #    data[key] = value.to(config.device)

            input_ids = input_ids.to(config.device)
            attention_mask = attention_mask.to(config.device)
            targets = targets.to(config.device)
            

            batch_preds, loss, metrics = model(input_ids, attention_mask,targets)
            fin_loss += loss.item()
            fin_preds.append(batch_preds.cpu().detach().numpy())

    
    return fin_preds, fin_loss / len(data_loader), metrics