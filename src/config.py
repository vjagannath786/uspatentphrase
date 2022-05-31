import transformers
from transformers import AutoTokenizer

path = '../'
lr = 2e-5
batch_size = 32
seed = 2021
device = 'cuda'
epochs = 1
train_file = '../train_folds.csv'
titles_file = '../titles.csv'
max_len = 64
num_workers = 2


deberata_model = 'microsoft/deberta-v3-small'
deberta_tokenizer = AutoTokenizer.from_pretrained(deberata_model, do_lower_case= True)