import transformers
from transformers import AutoTokenizer

path = '../'
lr = 2e-5
batch_size = 32
seed = 2021
device = 'cuda'
epochs = 5
train_file = '../train_folds.csv'
titles_file = '../titles.csv'
max_len = 200
num_workers = 1


deberata_model = 'microsoft/deberta-v3-small'
deberta_tokenizer = AutoTokenizer.from_pretrained(deberata_model, do_lower_case= True)