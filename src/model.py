import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
import config


def loss_fn(outputs, targets):

    #print(outputs)
    #print(targets)
    
    return torch.sqrt(nn.MSELoss()(outputs, targets))


class PhraseModel(nn.Module):
    def __init__(self, config, dropout):
        super(PhraseModel, self).__init__()
        self.deberta = AutoModel.from_pretrained('microsoft/deberta-v3-small', config=config)
        
        self.drop1 = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(768)
        self.l1 = nn.Linear(768,1)

    '''
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

    '''
    def forward(self,ids, mask, token_type_ids, score=None):
        _out = self.deberta(ids, mask, token_type_ids)

        #print('I am here')
        #print(_out)
        x = _out['hidden_states']
        #x = torch.cat((x[-1], x[-2]), dim=-1)
        x = x[-1]
        
        #x = self.layer_norm(x)
        
        #x = torch.mean(x,1, True)
        x = self.layer_norm(x)
        x = self.drop1(x)       
        #x = x.permute(0,2,1)
        #x = self.conv1(x)
        x = self.l1(x)
        #print(x.size())


        outputs =x.squeeze(-1)

        
        


        if score is None:
            
            return outputs
        else:
            
            loss = loss_fn(outputs, score.unsqueeze(1))
            return outputs, loss




