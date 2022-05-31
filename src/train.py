import numpy as np
import pandas as pd
import config
from dataset import PhraseDataset
from model import PhraseModel
from transformers import AutoConfig
import torch
from transformers import AdamW
from transformers import (get_linear_schedule_with_warmup,get_constant_schedule_with_warmup)
from pytorchtools import EarlyStopping
import engine
import random
from sklearn.metrics import mean_squared_error



roberta_pred = None

def _loss_fn(targets, outputs):
    #print(outputs.shape)
    #print(targets.unsqueeze(1).shape)
    
    #loss = 
    #print(loss.view(-1))
    #print((outputs))
    #print((targets))
    return mean_squared_error(targets, outputs, squared=False)

def run_training(df, i):

    train_fold = df.loc[df['kfold'] != i].reset_index(drop=True)
    valid_fold = df.loc[df['kfold'] == i].reset_index(drop=True)



    trainset = PhraseDataset(anchor= train_fold['anchor'].values, target= train_fold['target'].values,  title= train_fold['title'],score=train_fold['score'], tokenizer= config.deberta_tokenizer, max_len= config.max_len)
    validset = PhraseDataset(anchor= valid_fold['anchor'].values, target= valid_fold['target'].values,  title= valid_fold['title'],score=valid_fold['score'], tokenizer= config.deberta_tokenizer, max_len= config.max_len)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size = config.batch_size, num_workers = config.num_workers)
    validloader = torch.utils.data.DataLoader(validset, batch_size = config.batch_size, num_workers = config.num_workers)

    model_config = AutoConfig.from_pretrained('microsoft/deberta-v3-small')
    model_config.output_hidden_states = True

    model_config.return_dict = True

    model = PhraseModel(config= model_config, dropout=0.1)
    model.to(config.device)

    parameter_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]

    optimizer_parameters = [
        {
            "params": [
                p for n, p in parameter_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.001,
        },
        {
            "params": [
                p for n, p in parameter_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    num_train_steps = int(len(train_fold) / config.batch_size * 15)

    optimizer = AdamW(optimizer_parameters, lr=config.lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_train_steps)

    early_stopping = EarlyStopping(patience=4, path=f'../../working/checkpoint_deberta_{i}_v1.pt',verbose=True)


    best_loss = 1000
    
    for epoch in range(config.epochs):
        

        print('Epoch:', epoch,'LR:', scheduler.get_last_lr())
        train_loss = engine.train_fn(model, trainloader, optimizer, scheduler)
        valid_preds, valid_loss = engine.eval_fn(model, validloader)


        print(f'train_loss {train_loss} and valid_loss {valid_loss}')
        '''
        if valid_loss < best_loss:
            print('valid loss improved saving model')
            torch.save(model.state_dict(), f'model_{fold}.bin')
            best_loss = valid_loss
        '''
        #scheduler.step()

        early_stopping(valid_loss, model)
        
        if early_stopping.early_stop:
            print("Early stopping")
            break
        else:
            roberta_pred = valid_preds



        #model.load_state_dict(torch.load(f'checkpoint_{fold}.pt'))



    return roberta_pred


if __name__ == "__main__":
    df = pd.read_csv(config.train_file)

    df1 = pd.read_csv(config.titles_file)

    final_df = df.merge(df1, left_on='context', right_on='code', how='left')

    print(df.shape)

    print(final_df.shape)

    print(final_df[['anchor','target','context','title', 'score']])

    #_dataset = PhraseDataset(anchor= final_df['anchor'].values, target= final_df['target'].values,  title= final_df['title'],score=final_df['score'], tokenizer= config.deberta_tokenizer, max_len= 64)

    #print(_dataset[100])

    

    #model = PhraseModel(config=model_config, dropout=0.1)

    #data = _dataset[0]

    #trainloader = torch.utils.data.DataLoader(_dataset, batch_size = config.batch_size, num_workers = 2)
    #validloader = torch.utils.data.DataLoader(validset, batch_size = config.VALID_BATCH_SIZE, sampler=validsampler,num_workers = config.NUM_WORKERS)

    #for i in trainloader:
    #    outputs, loss = model(ids=i['ids'],mask=i['mask'], score= i['score'])
    #    break

    _outputs = []
    _targets = []
    

    for i in range(5):
        
        #df = pd.read_csv(config.train_file)
        tmp_target = final_df.query(f"kfold == {i}")['score'].values
        tmp = run_training(final_df, i)

        a = np.concatenate(tmp,axis=0)
        b = np.concatenate(a, axis=0)

        #print(len(b))
        #print(len(tmp_target))

        loss =  _loss_fn(tmp_target, b)

        print(f'loss for fold {i} is {loss}')

        #print(b)
        #print(tmp_target)
        _outputs.append(b)
        _targets.append(tmp_target)

    

    c = np.concatenate(_outputs, axis=0)
    d = np.concatenate(_targets,axis=0)
    total_loss =  _loss_fn(d,c)

    print(f'total loss for for all folds is {total_loss}')

