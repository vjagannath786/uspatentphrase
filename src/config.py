import transformers
from transformers import AutoTokenizer

path = '../'
lr = 1e-5
batch_size = 32
seed = 2021
device = 'cuda'
epochs = 2
train_file = '../train_folds.csv'
titles_file = '../titles.csv'
max_len = 150
num_workers = 2


deberata_model = "anferico/bert-for-patents"
#"microsoft/deberta-v3-base"
#'sentence-transformers/all-mpnet-base-v2'
#"cross-encoder/nli-deberta-v3-base"


model_config = "anferico/bert-for-patents"
#"microsoft/deberta-v3-base"
#
#"cross-encoder/nli-deberta-v3-base"

deberta_tokenizer = AutoTokenizer.from_pretrained(deberata_model, do_lower_case= True)