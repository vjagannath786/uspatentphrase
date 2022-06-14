import transformers
from transformers import AutoTokenizer

path = '../'
lr = 1e-5
batch_size = 32
seed = 2021
device = 'cuda'
epochs = 5
train_file = '../../input/us-patent-phrase-to-phrase-matching/train.csv'
titles_file = '../titles.csv'
max_len = 150
num_workers = 2


deberata_model = 'sentence-transformers/bert-base-nli-mean-tokens'

model_config = "sentence-transformers/bert-base-nli-mean-tokens"
deberta_tokenizer = AutoTokenizer.from_pretrained(deberata_model, do_lower_case= True)