import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler

'''
preprocess text to tranform tags in general @USER tags and urls to general HTTPURL tags
'''
def preprocess_text(post):
    new_post = ""
    for t in post.split(" "):
        t = '@USER' if t.startswith('@') and len(t) > 1 else t
        t = 'HTTPURL' if 'http' in t else t
        #for RumourEval and Contextual Abuse
        #t = 'HTTPURL' if 'http' in  if t.startswith('http') else t
        new_post = new_post + t + " "

    return new_post

'''
given a number, translate in a more "textual" description
'''
def generate_descriptive_number_from_int(number):
    if number == 1:
        return "1st"
    elif number == 2:
        return "2nd"
    elif number == 3:
        return "3rd"
    else:
        return "%dth" % number#check

'''
generate a textual description of the time interval
'''
def generate_descriptive_date_from_seconds(seconds):
    #compute seconds and minutes
    m, s = divmod(seconds // 1000, 60)
    #compute hours and minutes
    h, m = divmod(m, 60)
    #compute days and hours
    d, h = divmod(h, 24)

    #translate in a more textual description and return the time interval as descripted. Delete seconds information
    #(too much fine grained)
    m = str(m)
    h = str(h)
    d = str(d)

    return "after %s days, %s hours, %s minutes" % (d, h, m)

'''
Class to load the data for the SINGLE model
<s> target_post </s>
'''
class Data_SINGLE(Dataset):

    def __init__(self, data, tokenizer, max_len_bert):
        #load the given tokenizer
        self.tokenizer = tokenizer
        #load the maximum length of the input
        self.max_len_bert = max_len_bert

        self.input_ids = []
        self.attention_masks = []
        self.text_posts = []
        self.decoded = []
        self.labels = []

        for idx in range(len(data['target_label'])):
            #get the target post
            dial = data['dialogue'][idx]
            text_post = preprocess_text(dial[-1])
            #encode the post
            encoding = self.tokenizer.encode_plus(
                text_post,
                max_length=self.max_len_bert,
                add_special_tokens=True,
                padding='max_length',
                return_attention_mask=True,
                return_token_type_ids=False,
                return_tensors='pt',
                truncation=True
            )
            #save the encoding
            self.input_ids.append(encoding['input_ids'])
            self.attention_masks.append(encoding['attention_mask'])
            self.text_posts.append(text_post)
            self.decoded.append(self.tokenizer.convert_ids_to_tokens(encoding['input_ids'].flatten()))
            self.labels.append(data['target_label'][idx])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        return {
            'text': self.text_posts[idx],
            'decoded': self.decoded[idx],
            'input_ids': self.input_ids[idx].flatten(),
            'attention_mask': self.attention_masks[idx].flatten(),
            'labels': torch.Tensor([int(self.labels[idx])])
        }

'''
DataLoader for RoBERTa-S input format, half training set per epoch
'''
def create_data_loaderData_SINGLE(data, tokenizer, max_len_bert, batch_size, eval=False):
    ds = Data_SINGLE(
        data=data,
        tokenizer=tokenizer,
        max_len_bert=max_len_bert
    )
    if not eval:
        return DataLoader(ds, batch_size=batch_size, num_workers=4,
                          sampler=RandomSampler(ds, num_samples=len(ds)//2, replacement=False, generator=None)), len(ds)//2

    else:
        return DataLoader(ds, batch_size=batch_size, num_workers=4, shuffle=False), len(ds)


'''
Class to load the data in the format "RoBERTa-P"
<s> compared_post </s></s> target_post </s>
where compared_post can be target-1, or the root of the chain
'''
class Data_PAIR(Dataset):

    def __init__(self, data, tokenizer, max_len_bert):
        #load the given tokenizer
        self.tokenizer = tokenizer
        #load the maximum length of the input
        self.max_len_bert = max_len_bert

        self.input_ids = []
        self.attention_masks = []
        self.text_posts = []
        self.decoded = []
        self.labels = []

        for idx in range(len(data['target_label'])):
            #get the target post
            dial = data['dialogue'][idx]
            text_post = preprocess_text(dial[-1])
            #get the compared post
            compared_post = preprocess_text(dial[-2])
            #encode the post
            encoding = self.tokenizer.encode_plus(
                compared_post,
                text_post,
                max_length=self.max_len_bert,
                add_special_tokens=True,
                padding='max_length',
                return_attention_mask=True,
                return_token_type_ids=False,
                return_tensors='pt',
                truncation=True
            )
            #save the encoding
            self.input_ids.append(encoding['input_ids'])
            self.attention_masks.append(encoding['attention_mask'])
            self.text_posts.append(text_post)
            self.decoded.append(self.tokenizer.convert_ids_to_tokens(encoding['input_ids'].flatten()))
            self.labels.append(data['target_label'][idx])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        return {
            'text': self.text_posts[idx],
            'decoded': self.decoded[idx],
            'input_ids': self.input_ids[idx].flatten(),
            'attention_mask': self.attention_masks[idx].flatten(),
            'labels': torch.Tensor([int(self.labels[idx])])
        }

'''
DataLoader for RoBERTa-P input format, half training set per epoch
'''
def create_data_loaderData_PAIR(data, tokenizer, max_len_bert, batch_size, eval=False):
    ds = Data_PAIR(
        data=data,
        tokenizer=tokenizer,
        max_len_bert=max_len_bert
    )
    if not eval:
        return DataLoader(ds, batch_size=batch_size, num_workers=4, sampler=RandomSampler(ds, num_samples=len(ds)//2, replacement=False, generator=None)), len(ds)//2

    else:
        return DataLoader(ds, batch_size=batch_size, num_workers=4, shuffle=False), len(ds)