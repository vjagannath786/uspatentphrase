import more_itertools
import multiprocessing
from torch.utils.data import Sampler, Dataset, DataLoader
from scipy import stats
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
from scipy import stats



roberta_pred = None

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def _loss_fn(targets, outputs):
    #print(outputs.shape)
    #print(targets.unsqueeze(1).shape)
    
    #loss = 
    #print(loss.view(-1))
    #print((outputs))
    #print((targets))
    return mean_squared_error(targets, outputs, squared=True)


def monitor_metrics(outputs, targets):
        #device = targets.get_device()
        #outputs = outputs.cpu().detach().numpy().ravel()
        #targets = targets.cpu().detach().numpy().ravel()
        #print(outputs)
        #print(targets)
        pearsonr = stats.pearsonr(outputs, targets)
        return {"pearsonr": torch.tensor(pearsonr[0], device=config.device)}


def compute_pearson(outputs, labels):
    # Squash values between 0 to 1
    outputs[outputs < 0] = 0
    outputs[outputs > 1] = 1
    
    # Round off to nearest 0.25 factor
    outputs = 0.25 * np.round(outputs/0.25) 
    
    pearsonr = stats.pearsonr(outputs, labels)[0]
    return pearsonr


class SmartBatchingDataset(Dataset):
    def __init__(self, df, tokenizer):
        super(SmartBatchingDataset, self).__init__()
        self._data = (
            df.title + f"[{df.section}]" + df.anchor + f"[{df.section}]" + df.target
        ).apply(tokenizer.tokenize).apply(tokenizer.convert_tokens_to_ids).to_list()
        self._targets = None
        if 'score' in df.columns:
            self._targets = df.score.tolist()
        self.sampler = None

    def __len__(self):
        return len(self._data)

    def __getitem__(self, item):
        if self._targets is not None:
            return self._data[item], self._targets[item]
        else:
            return self._data[item]

    def get_dataloader(self, batch_size, max_len, pad_id):
        self.sampler = SmartBatchingSampler(
            data_source=self._data,
            batch_size=batch_size
        )
        collate_fn = SmartBatchingCollate(
            targets=self._targets,
            max_length=max_len,
            pad_token_id=pad_id
        )
        dataloader = DataLoader(
            dataset=self,
            batch_size=batch_size,
            sampler=self.sampler,
            collate_fn=collate_fn,
            num_workers=(multiprocessing.cpu_count()-1),
            pin_memory=True
        )
        return dataloader

class SmartBatchingSampler(Sampler):
    def __init__(self, data_source, batch_size):
        super(SmartBatchingSampler, self).__init__(data_source)
        self.len = len(data_source)
        sample_lengths = [len(seq) for seq in data_source]
        argsort_inds = np.argsort(sample_lengths)
        self.batches = list(more_itertools.chunked(argsort_inds, n=batch_size))
        self._backsort_inds = None
    
    def __iter__(self):
        if self.batches:
            last_batch = self.batches.pop(-1)
            np.random.shuffle(self.batches)
            self.batches.append(last_batch)
        self._inds = list(more_itertools.flatten(self.batches))
        yield from self._inds

    def __len__(self):
        return self.len
    
    @property
    def backsort_inds(self):
        if self._backsort_inds is None:
            self._backsort_inds = np.argsort(self._inds)
        return self._backsort_inds


class SmartBatchingCollate:
    def __init__(self, targets, max_length, pad_token_id):
        self._targets = targets
        self._max_length = max_length
        self._pad_token_id = pad_token_id
        
    def __call__(self, batch):
        if self._targets is not None:
            sequences, targets = list(zip(*batch))
        else:
            sequences = list(batch)
        
        input_ids, attention_mask = self.pad_sequence(
            sequences,
            max_sequence_length=self._max_length,
            pad_token_id=self._pad_token_id
        )
        
        if self._targets is not None:
            output = input_ids, attention_mask, torch.tensor(targets)
        else:
            output = input_ids, attention_mask
        return output
    
    def pad_sequence(self, sequence_batch, max_sequence_length, pad_token_id):
        max_batch_len = max(len(sequence) for sequence in sequence_batch)
        max_len = min(max_batch_len, max_sequence_length)
        padded_sequences, attention_masks = [[] for i in range(2)]
        attend, no_attend = 1, 0
        for sequence in sequence_batch:
            # As discussed above, truncate if exceeds max_len
            new_sequence = list(sequence[:max_len])
            
            attention_mask = [attend] * len(new_sequence)
            pad_length = max_len - len(new_sequence)
            
            new_sequence.extend([pad_token_id] * pad_length)
            attention_mask.extend([no_attend] * pad_length)
            
            padded_sequences.append(new_sequence)
            attention_masks.append(attention_mask)
        
        padded_sequences = torch.tensor(padded_sequences)
        attention_masks = torch.tensor(attention_masks)
        return padded_sequences, attention_masks

def run_training(df, i):

    
    print(df['kfold'].unique())

    train_fold = df.loc[df['kfold'] != i].reset_index(drop=True)
    valid_fold = df.loc[df['kfold'] == i].reset_index(drop=True)



    #trainset = PhraseDataset(title= train_fold['title'].values,anchor= train_fold['anchor'].values,target= train_fold['target'].values,
    #section=train_fold['section'].values,score=train_fold['score'], tokenizer= config.deberta_tokenizer, max_len= config.max_len)
    #validset = PhraseDataset(title= valid_fold['title'].values,anchor= valid_fold['anchor'].values,target= valid_fold['target'].values,
    #section=valid_fold['section'].values,score=valid_fold['score'], tokenizer= config.deberta_tokenizer, max_len= config.max_len)

    #trainloader = torch.utils.data.DataLoader(trainset, batch_size = config.batch_size, num_workers = config.num_workers)
    #validloader = torch.utils.data.DataLoader(validset, batch_size = config.batch_size, num_workers = config.num_workers)

    
    trainset = SmartBatchingDataset(train_fold, config.deberta_tokenizer)
    trainloader = trainset.get_dataloader(batch_size=config.batch_size, max_len=config.max_len, pad_id=config.deberta_tokenizer.pad_token_id)

    validset = SmartBatchingDataset(valid_fold, config.deberta_tokenizer)
    validloader = validset.get_dataloader(batch_size=config.batch_size, max_len=config.max_len, pad_id=config.deberta_tokenizer.pad_token_id)

    model_config = AutoConfig.from_pretrained(config.model_config)
    model_config.output_hidden_states = True

    model_config.return_dict = True
    model_config.attention_probs_dropout_prob = 0

    model = PhraseModel(_config= model_config, dropout=0.1)
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

    optimizer = torch.optim.AdamW(optimizer_parameters, lr=config.lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_train_steps)

    early_stopping = EarlyStopping(patience=4, path=f'../../working/checkpoint_deberta_{i}_v1.pt',verbose=True)


    best_loss = 1000
    
    for epoch in range(config.epochs):
        

        print('Epoch:', epoch,'LR:', scheduler.get_last_lr())
        train_loss, train_metrics = engine.train_fn(model, trainloader, optimizer, scheduler)
        valid_preds, valid_loss, valid_metrics = engine.eval_fn(model, validloader)


        print(f'train_loss {train_loss} and valid_loss {valid_loss}')

        print(f'correlation for train {train_metrics} and valid correlation {valid_metrics}')
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

    seed_everything(config.seed)
    df = pd.read_csv(config.train_file)

    df1 = pd.read_csv(config.titles_file)

    final_df = df.merge(df1, left_on='context', right_on='code', how='left')

    print(df.shape)

    print(final_df.shape)

    print(final_df[['anchor','target','context','title', 'score']])

    final_df['text'] = final_df['context'] + '[SEP]' + final_df['target'] + '[SEP]'  + final_df['anchor']

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
        metrics = monitor_metrics(b, tmp_target)

        print(f'loss for fold {i} is {loss}')

        print(f'pearson is {metrics}')

        #print(b)
        #print(tmp_target)
        _outputs.append(b)
        _targets.append(tmp_target)

    

    c = np.concatenate(_outputs, axis=0)
    d = np.concatenate(_targets,axis=0)
    total_loss =  _loss_fn(d,c)

    pearson = compute_pearson(d, c)

    print(f'total loss for for all folds is {total_loss}')

    print(f'correlation coef is {pearson}')

