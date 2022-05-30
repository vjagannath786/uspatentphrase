import pandas as pd
import config
from dataset import PhraseDataset
from model import PhraseModel
from transformers import AutoConfig



if __name__ == "__main__":
    df = pd.read_csv(config.train_file)

    df1 = pd.read_csv(config.titles_file)

    final_df = df.merge(df1, left_on='context', right_on='code', how='left')

    print(df.shape)

    print(final_df.shape)

    print(final_df[['anchor','target','context','title', 'score']])

    _dataset = PhraseDataset(anchor= final_df['anchor'].values, target= final_df['target'].values,  title= final_df['title'],score=final_df['score'], tokenizer= config.deberta_tokenizer, max_len= 200)

    print(_dataset[100])

    model_config = AutoConfig.from_pretrained('../../input/debertalarge')
    model_config.output_hidden_states = True

    model = PhraseModel(config=model_config, dropout=0.1)

    data = _dataset[0]

    
    
    outputs, loss = model(**data)

