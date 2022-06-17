import more_itertools
import multiprocessing
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoModelForSequenceClassification
from config import model_config
import config
from dataset import PhraseDataset
import pandas as pd
from transformers import AutoConfig
from scipy.spatial.distance import cosine
import random
import numpy as np
from scipy import stats
from torch.utils.data import Sampler, Dataset, DataLoader



def loss_fn(outputs, targets):

    #print(outputs)
    #print(targets)
    
    return nn.MSELoss()(outputs, targets)


def contrastive_loss(output1, output2, label, margin=1.0):
    euclidean_distance = F.pairwise_distance(output1, output2)
    loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                  (label) * torch.pow(F.relu(margin - euclidean_distance), 2))
    return loss_contrastive




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







class AttentionHead(nn.Module):
    def __init__(self, in_size: int = 1024, hidden_size: int = 512) -> None:
        super().__init__()
        self.W = nn.Linear(in_size, hidden_size)
        self.V = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(0.3)

    def forward(self, features):
        att = torch.tanh(self.W(features))
        score = self.V(att)
        attention_weights = torch.softmax(score, dim=1)
        context_vector = attention_weights * features
        context_vector = torch.sum(context_vector, dim=1)
        output = self.dropout(context_vector)
        return output


def monitor_metrics(outputs, targets):
        device = targets.get_device()
        outputs = outputs.cpu().detach().numpy().ravel()
        targets = targets.cpu().detach().numpy().ravel()
        #print(outputs)
        #print(targets)
        pearsonr = stats.pearsonr(outputs, targets)
        return {"pearsonr": torch.tensor(pearsonr[0], device=config.device)}


class PhraseModel(nn.Module):
    def __init__(self, _config, dropout):
        super(PhraseModel, self).__init__()
        self.deberta = AutoModelForSequenceClassification.from_pretrained(config.model_config, config=_config)
        #self.deberta1 = AutoModel.from_pretrained('princeton-nlp/unsup-simcse-bert-base-uncased', config=_config)
        #self.deberta2 = AutoModel.from_pretrained('princeton-nlp/unsup-simcse-bert-base-uncased', config=_config)

        #self.deberta1 = AutoModel.from_pretrained('../../input/debertalarge', config=config)
        
        self.drop1 = nn.Dropout(dropout)
        #self.layer_norm = nn.LayerNorm(1024)

        self.cosine = nn.CosineSimilarity(dim=-1)

        self.l1 = nn.Linear(1024,1)

        self._init_weights(self.l1)

        self.attention = AttentionHead()

        #self.attention = nn.Sequential(
        #    nn.Linear(1024, 512),
        #    nn.Tanh(),
        #    nn.Linear(512, 1),
        #    nn.Softmax(dim=1)
        #)

        self._init_weights(self.attention)
        #self.weights_init_custom()

    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            print('in init')
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def weights_init_custom(self):
        init_layers = [23, 22, 21,20,19]
        dense_names = ["query", "key", "value", "dense"]
        layernorm_names = ["LayerNorm"]
        for name, module in self.deberta.named_parameters():
            if any(f".{i}." in name for i in init_layers):
                if any(n in name for n in dense_names):
                    if "bias" in name:
                        module.data.zero_()
                    elif "weight" in name:
                        print(name)
                        module.data.normal_(mean=0.0, std=0.02)
                elif any(n in name for n in layernorm_names):
                    if "bias" in name:
                        module.data.zero_()
                    elif "weight" in name:
                        module.data.fill_(1.0)

    '''
    def feature(self, ids, mask, token_type_ids):
        outputs = self.deberta(ids, mask, token_type_ids)
        last_hidden_states = outputs[0]
        # feature = torch.mean(last_hidden_states, 1)
        weights = self.attention(last_hidden_states)
        feature = torch.sum(weights * last_hidden_states, dim=1)
        return feature
    '''

    
    def forward(self,ids, mask, targets=None):
        _out = self.deberta(ids, mask)
        #_out1 = self.deberta1(ids1, mask1, token_type_ids1)
        #_out2 = self.deberta2(ids2, mask2, token_type_ids2)

        #print(_out)

        #hidden_states = _out[1]
        #pooled_output = torch.cat(tuple([hidden_states[i] for i in [-4, -3, -2, -1]]), dim=-1)
        #pooled_output = pooled_output[:, 0, :]



        #print(pooled_output)
        #print('I am here')
        #print(_out)
        #x = pooled_output
        #x = torch.cat((x[-1], x[-2], x[-3], x[-4]), dim=-1)
        x = _out['hidden_states']
        #x1 = _out1['hidden_states']
        #x2 = _out2['hidden_states']

        

        x = x[-1]
        #x1 = x1[-1]
        #x2 = x2[-1]

        x = self.attention(x)
        #x1 = self.attention(x1)
        #x2 = self.attention(x2)


        #x = torch.cat((x[-1]), dim=-1)
        #x1 = torch.cat((x2[-1]), dim=-1)
        #2 = torch.cat((x2[-1]), dim=-1)

        #x = torch.mean(x,1, True)
        #x1 = torch.mean(x1,1, True)
        #x2 = torch.mean(x2,1, True)
        
        #x = self.layer_norm(x)
        
        #x = torch.mean(x,1, True)
        #x = self.layer_norm(x)
        #x = self.drop1(x)       
        #x = x.permute(0,2,1)
        #x = self.conv1(x)

        #x1 = _out1['hidden_states']
        #x1 = torch.cat((x1[-1], x1[-2], x1[-3], x1[-4]), dim=-1)
        #x1 = torch.mean(x1,1, True)
        #x1 = self.drop1(x1)


        #x3 = self.cosine(x,x1)

        #cosine_sim_0_1 = 1-self.cosine(x, x1)
        #cosine_sim_0_2 = 1-self.cosine(x, x2)

        #print(cosine_sim_0_1)
        #print(cosine_sim_0_2)

        #x3 = torch.cat([cosine_sim_0_1, cosine_sim_0_2], dim=-1)

        #x3 = torch.dist(cosine_sim_0_1, cosine_sim_0_2,2)
        #x3 = torch.abs(cosine_sim_0_2 - cosine_sim_0_1)
        #x3 = F.pairwise_distance(cosine_sim_0_1, cosine_sim_0_2, keepdim=False)

        #print(x3)
        x = self.drop1(x)
        x4 = self.l1(x)
        #print(x.size())

        #print(x4)

        outputs = x4

        
        


        if targets is None:
            
            return outputs
        else:
            
            loss = loss_fn(outputs, targets.unsqueeze(1))

            #loss =  contrastive_loss(cosine_sim_0_1, cosine_sim_0_2, score.unsqueeze(1))
            
            metrics = monitor_metrics(outputs, targets.unsqueeze(1))
            #print(loss)
            #print(metrics)
            
            return outputs, loss, metrics
    '''

    def forward(self,ids, mask, token_type_ids, score=None):
        feature = self.feature(ids, mask, token_type_ids)
        output = self.l1(self.drop1(feature))
        loss = loss_fn(output, score.unsqueeze(1))
        return output, loss
    '''

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


if __name__ == "__main__":

    seed_everything(config.seed)

    df = pd.read_csv(config.train_file)

    

    df1 = pd.read_csv(config.titles_file)

    final_df = df.merge(df1, left_on='context', right_on='code', how='left')

    final_df = final_df.reset_index(drop=True)

    final_df['text'] = final_df['context'] + '[SEP]' + final_df['target'] + '[SEP]'  + final_df['anchor']

    #_dataset = PhraseDataset(title= final_df['title'].values,anchor= final_df['anchor'].values,target= final_df['target'].values,
    #                        section=final_df['section'].values,
    #                        score=final_df['score'], tokenizer= config.deberta_tokenizer, max_len= config.max_len)


    #data = _dataset
    #collate = Collate(tokenizer=config.deberta_tokenizer)

    #trainloader = torch.utils.data.DataLoader(data, batch_size = config.batch_size, num_workers = config.num_workers, collate_fn=collate)

    dataset = SmartBatchingDataset(final_df, config.deberta_tokenizer)
    dataloader = dataset.get_dataloader(batch_size=32, max_len=150, pad_id=config.deberta_tokenizer.pad_token_id)

    model_config = AutoConfig.from_pretrained(config.model_config)
    model_config.output_hidden_states = True
    model_config.attention_probs_dropout_prob = 0

    #model_config.return_dict = True

    model  = PhraseModel(_config=model_config, dropout=0.2)

    for i, (input_ids, attention_mask, targets) in enumerate(dataloader):
        print(i)
        output, loss, metrics = model(input_ids, attention_mask, targets)
        print(loss)
        print(metrics)
        break




