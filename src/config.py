import transformers
from transformers import AutoTokenizer

path = '../'
lr = 4e-5
batch_size = 1
seed = 2021
device = 'cuda'
epochs = 1
train_file = '../train_folds.csv'
titles_file = '../titles.csv'
max_len = 150
num_workers = 2


deberata_model = 'microsoft/deberta-v3-large'

model_config = "../../input/deberta-v3-large/deberta-v3-large"
deberta_tokenizer = AutoTokenizer.from_pretrained(deberata_model, do_lower_case= True)