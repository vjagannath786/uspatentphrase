import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from config import model_config
import config
from dataset import PhraseDataset
import pandas as pd
from transformers import AutoConfig


def loss_fn(outputs, targets):

    #print(outputs)
    #print(targets)
    
    return torch.sqrt(nn.MSELoss()(outputs, targets))





class PhraseModel(nn.Module):
    def __init__(self, config, dropout):
        super(PhraseModel, self).__init__()
        self.deberta = AutoModel.from_pretrained('microsoft/deberta-v3-large', config=config)

        #self.deberta1 = AutoModel.from_pretrained('../../input/debertalarge', config=config)
        
        self.drop1 = nn.Dropout(dropout)
        #self.layer_norm = nn.LayerNorm(1024)

        #self.cosine = nn.CosineSimilarity(dim=-1)

        self.l1 = nn.Linear(1024,1)

        self._init_weights(self.l1)

        self.attention = nn.Sequential(
            nn.Linear(1024, 512),
            nn.Tanh(),
            nn.Linear(512, 1),
            nn.Softmax(dim=1)
        )

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

    def feature(self, ids, mask, token_type_ids):
        outputs = self.deberta(ids, mask, token_type_ids)
        last_hidden_states = outputs[0]
        # feature = torch.mean(last_hidden_states, 1)
        weights = self.attention(last_hidden_states)
        feature = torch.sum(weights * last_hidden_states, dim=1)
        return feature

    '''
    def forward(self,ids, mask, token_type_ids, ids1, mask1, token_type_ids1,score=None):
        _out = self.deberta(ids, mask, token_type_ids)
        _out1 = self.deberta1(ids1, mask1, token_type_ids1)




        #print('I am here')
        #print(_out)
        x = _out['hidden_states']
        x = torch.cat((x[-1], x[-2], x[-3], x[-4]), dim=-1)
        #x = x[-1]
        
        #x = self.layer_norm(x)
        
        x = torch.mean(x,1, True)
        #x = self.layer_norm(x)
        #x = self.drop1(x)       
        #x = x.permute(0,2,1)
        #x = self.conv1(x)

        #x1 = _out1['hidden_states']
        #x1 = torch.cat((x1[-1], x1[-2], x1[-3], x1[-4]), dim=-1)
        #x1 = torch.mean(x1,1, True)
        #x1 = self.drop1(x1)


        #x3 = self.cosine(x,x1)


        #print(x3)
        x4 = self.l1(x)
        #print(x.size())

        outputs =x4

        
        


        if score is None:
            
            return outputs
        else:
            
            loss = loss_fn(outputs, score.unsqueeze(1))
            
            return outputs, loss
    '''

    def forward(self,ids, mask, token_type_ids, score=None):
        feature = self.feature(ids, mask, token_type_ids)
        output = self.l1(self.drop1(feature))
        loss = loss_fn(output, score.unsqueeze(1))
        return output, loss




if __name__ == "__main__":
    df = pd.read_csv(config.train_file)

    df1 = pd.read_csv(config.titles_file)

    final_df = df.merge(df1, left_on='context', right_on='code', how='left')

    final_df = final_df.reset_index(drop=True)

    _dataset = PhraseDataset(anchor= final_df['anchor'].values, target= final_df['target'].values,  
    title= final_df['title'],score=final_df['score'], tokenizer= config.deberta_tokenizer, max_len= config.max_len)


    data = _dataset

    trainloader = torch.utils.data.DataLoader(data, batch_size = config.batch_size, num_workers = config.num_workers)

    model_config = AutoConfig.from_pretrained(config.model_config)
    model_config.output_hidden_states = True

    #model_config.return_dict = True

    model  = PhraseModel(config=model_config, dropout=0.1)

    for i in trainloader:
        output, loss = model(**i)
        print(loss)
        break




