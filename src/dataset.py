import torch

class PhraseDataset:
    def __init__(self, anchor,title,target, section,score, tokenizer, max_len):
        self.anchor = anchor
        self.target = target
        self.title = title
        self.section = section
        self.score = score
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.anchor)

    def __getitem__(self, item):

        target = self.target[item]
        anchor = self.anchor[item]
        title = self.title[item]
        section = str(self.section[item]).lower()
        score = self.score[item]

        encoded_text = self.tokenizer.encode_plus(
            str(title).lower() + f"[{section}]" + str(anchor).lower() + f"[{section}]" + str(target).lower(),
            padding="max_length",
            max_length=self.max_len,
            truncation=True,
            return_token_type_ids=True,
        )

        '''
        encoded_text1 = self.tokenizer.encode_plus(
            anchor,
            padding="max_length",
            max_length=60,
            truncation=True,
            return_token_type_ids=True,
        )

        encoded_text2 = self.tokenizer.encode_plus(
            target,
            padding="max_length",
            max_length=60,
            truncation=True,
            return_token_type_ids=True,
        )
        '''


        input_ids = encoded_text["input_ids"]
        attention_mask = encoded_text["attention_mask"]
        token_type_ids = encoded_text["token_type_ids"]

        '''
        input_ids1 = encoded_text1["input_ids"]
        attention_mask1 = encoded_text1["attention_mask"]
        token_type_ids1 = encoded_text1["token_type_ids"]

        input_ids2 = encoded_text2["input_ids"]
        attention_mask2 = encoded_text2["attention_mask"]
        token_type_ids2 = encoded_text2["token_type_ids"]
        '''

        return {
            "ids": torch.tensor(input_ids, dtype=torch.long),
            "mask": torch.tensor(attention_mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            #"ids1": torch.tensor(input_ids1, dtype=torch.long),
            #"mask1": torch.tensor(attention_mask1, dtype=torch.long),
            #"token_type_ids1": torch.tensor(token_type_ids1, dtype=torch.long),
            #"ids2": torch.tensor(input_ids2, dtype=torch.long),
            #"mask2": torch.tensor(attention_mask2, dtype=torch.long),
            #"token_type_ids2": torch.tensor(token_type_ids2, dtype=torch.long),
            "score": torch.tensor(score, dtype=torch.float),
        }