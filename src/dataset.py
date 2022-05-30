import torch

class PhraseDataset:
    def __init__(self, anchor, target,  title,score, tokenizer, max_len):
        self.anchor = anchor
        self.target = target
        
        self.title = title
        self.score = score
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.anchor)

    def __getitem__(self, item):
        anchor = self.anchor[item]
        
        target = self.target[item]
        title = self.title[item]
        score = self.score[item]

        encoded_text = self.tokenizer.encode_plus(
            title +" " + anchor,
            target,
            padding="max_length",
            max_length=self.max_len,
            truncation=True,
        )
        input_ids = encoded_text["input_ids"]
        attention_mask = encoded_text["attention_mask"]
        token_type_ids = encoded_text["token_type_ids"]

        return {
            "ids": torch.tensor(input_ids, dtype=torch.long),
            "mask": torch.tensor(attention_mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "score": torch.tensor(score, dtype=torch.float),
        }