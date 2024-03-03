import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler
import copy

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
Class to load the data for the TC model
<s> root_post </s></s> ... </s></s> target-1 </s></s> target_post </s>
'''
class Data_TC(Dataset):
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
            # concatenate each message, adding the special tokens for start-end sentences
            chain = ""
            text_post = preprocess_text(dial[-1])

            for idx_d in range(len(dial)-1):
                chain = chain + preprocess_text(dial[idx_d]) + " </s></s> "
            # remove the last " </s></s> ", 10 character length
            chain = chain[:-10] + " "

            #encode the post
            encoding = self.tokenizer.encode_plus(
                chain,
                text_post,
                max_length=self.max_len_bert,
                add_special_tokens=True,
                padding='max_length',
                return_attention_mask=True,
                return_token_type_ids=False,
                return_tensors='pt',
                truncation=False
            )
            # if the resulting input is longer than the maximum length, remove sentences from the beginning of the chain,
            # keeping the root as first sentence of the chain
            if encoding['input_ids'].flatten().shape[0] > self.max_len_bert:

                flag = True
                i = 1

                while (flag):
                    chain = ""
                    for idx_d in range(len(dial)-1):
                        if idx_d == 0 or idx_d == len(dial)-2 or idx_d > i:
                            chain = chain + preprocess_text(dial[idx_d]) + " </s></s> "

                    chain = chain[:-10] + " "

                    encoding = self.tokenizer.encode_plus(
                        chain,
                        text_post,
                        max_length=self.max_len_bert,
                        add_special_tokens=True,
                        padding='max_length',
                        return_attention_mask=True,
                        return_token_type_ids=False,
                        return_tensors='pt',
                        truncation=False
                    )

                    if encoding['input_ids'].flatten().shape[0] <= self.max_len_bert:
                        flag = False

                    else:
                        # extreme case: if the only remained elements are the first and last one, truncate
                        if i >= len(dial) - 3:
                            chain = preprocess_text(dial[-2])
                            encoding = self.tokenizer.encode_plus(
                                chain,
                                text_post,
                                max_length=self.max_len_bert,
                                add_special_tokens=True,
                                padding='max_length',
                                return_attention_mask=True,
                                return_token_type_ids=False,
                                return_tensors='pt',
                                truncation='only_first'
                            )

                            flag = False
                        else:
                            i = i + 1


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
DataLoader for TC model, half training set per epoch
'''
def create_data_loaderData_TC(data, tokenizer, max_len_bert, batch_size, eval=False):
    ds = Data_TC(
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

Class to load the data for the TC + T model
<s> contextual_root_claim </s></s> ... </s></s> countextual_target_claim-1 </s></s> contextual_target_claim </s>
where a contextual post is a post augmented with metadata (e.g. author, time, etc.) in this way:
contextual_post = <t> time </t> post 

time is described in the following way: "after a days, b hours and c minutes"
'''
class Data_TC_T(Dataset):

    def __init__(self, data, tokenizer, max_len_bert):
        # load the given tokenizer
        self.tokenizer = tokenizer
        # load the maximum length of the input
        self.max_len_bert = max_len_bert

        self.input_ids = []
        self.attention_masks = []
        self.text_posts = []
        self.decoded = []
        self.labels = []

        for idx in range(len(data['target_label'])):
            # get dialogue, author sequence, parent sequence and time sequence
            dial = data['dialogue'][idx]
            time = data['time'][idx]

            chain = ""

            for idx_d in range(len(dial) - 1):
                # concatenate iteratively contextual posts in chronological order
                chain = chain + "<t> " + generate_descriptive_date_from_seconds(
                    time[idx_d]) + " </t> " + preprocess_text(dial[idx_d]) + " </s></s> "

            # remove last </s></s>
            chain = chain[:-10] + " "

            text_post = "<t> " + generate_descriptive_date_from_seconds(time[-1]) + " </t> " + preprocess_text(dial[-1])

            encoding = self.tokenizer.encode_plus(
                chain,
                text_post,
                max_length=self.max_len_bert,
                add_special_tokens=True,
                padding='max_length',
                return_attention_mask=True,
                return_token_type_ids=False,
                return_tensors='pt',
                truncation=False
            )
            # if the input is too long, remove the iteratively the second post until the input is not too long (keep the root)
            if encoding['input_ids'].flatten().shape[0] > self.max_len_bert:

                flag = True
                i = 1

                while (flag):
                    chain = ""
                    for idx_d in range(len(dial) - 1):
                        if idx_d == 0 or idx_d == len(dial) - 2 or idx_d > i:
                            chain = chain + "<t> " + generate_descriptive_date_from_seconds(
                                time[idx_d]) + " </t> " + preprocess_text(
                                dial[idx_d]) + " </s></s> "

                    chain = chain[:-10] + " "

                    encoding = self.tokenizer.encode_plus(
                        chain,
                        text_post,
                        max_length=self.max_len_bert,
                        add_special_tokens=True,
                        padding='max_length',
                        return_attention_mask=True,
                        return_token_type_ids=False,
                        return_tensors='pt',
                        truncation=False
                    )

                    if encoding['input_ids'].flatten().shape[0] <= self.max_len_bert:
                        flag = False
                    else:
                        # extreme case: if the only remained elements are the first and last one, truncate
                        if i >= len(dial) - 3:
                            chain = "<t> " + generate_descriptive_date_from_seconds(time[-2]) + " </t> " + \
                                    preprocess_text(dial[-2])

                            encoding = self.tokenizer.encode_plus(
                                chain,
                                text_post,
                                max_length=self.max_len_bert,
                                add_special_tokens=True,
                                padding='max_length',
                                return_attention_mask=True,
                                return_token_type_ids=False,
                                return_tensors='pt',
                                truncation='only_first'
                            )
                            flag = False
                        else:
                            i = i + 1

            # save the encoding
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
DataLoader for TC + T model, half training set per epoch
'''
def create_data_loaderData_TC_T(data, tokenizer, max_len_bert, batch_size, eval=False):
    ds = Data_TC_T(
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
Class to load the data for the TC + U model
<s> contextual_initial_claim </s></s> ... </s></s> contextual_target_claim-1 </s></s> contextual_target_claim </s>
where a contextual post is a post augmented with metadata (e.g. author, time, etc.) in this way:
contextual_post = <o> author_id </o> post 

author_id and parent_author_id are described in the following way: "x^th user"

The ids of authors and parents are unique inside each dialogue
'''


class Data_TC_U(Dataset):

    def __init__(self, data, tokenizer, max_len_bert):
        # load the given tokenizer
        self.tokenizer = tokenizer
        # load the maximum length of the input
        self.max_len_bert = max_len_bert

        self.input_ids = []
        self.attention_masks = []
        self.text_posts = []
        self.decoded = []
        self.labels = []

        for idx in range(len(data['target_label'])):
            # get dialogue, author sequence, parent sequence and time sequence
            dial = data['dialogue'][idx]
            orig = data['author_id'][idx]
            # save users in the dialogue in a list. The index is the local ID of the user
            list_user = list()

            chain = ""

            for idx_d in range(len(dial) - 1):
                if orig[idx_d] not in list_user:
                    list_user.append(orig[idx_d])
                # concatenate iteratively contextual posts in chronological order
                chain = chain + "<o> " + generate_descriptive_number_from_int(
                    list_user.index(orig[idx_d])) + " user </o> " + preprocess_text(dial[idx_d]) + " </s></s> "

            # remove last </s></s>
            chain = chain[:-10] + " "

            if orig[-1] not in list_user:
                list_user.append(orig[-1])
            text_post = "<o> " + generate_descriptive_number_from_int(list_user.index(orig[-1])) + " user </o> " + \
                        preprocess_text(dial[-1])

            encoding = self.tokenizer.encode_plus(
                chain,
                text_post,
                max_length=self.max_len_bert,
                add_special_tokens=True,
                padding='max_length',
                return_attention_mask=True,
                return_token_type_ids=False,
                return_tensors='pt',
                truncation=False
            )
            # if the input is too long, remove iteratively the first post until the input is not too long (except the initial claim)
            if encoding['input_ids'].flatten().shape[0] > self.max_len_bert:

                flag = True
                i = 1

                while (flag):
                    chain = ""
                    for idx_d in range(len(dial) - 1):
                        if idx_d == 0 or idx_d == len(dial) - 2 or idx_d > i:
                            chain = chain + "<o> " + generate_descriptive_number_from_int(
                                list_user.index(orig[idx_d])) + " user </o> " + preprocess_text(
                                dial[idx_d]) + " </s></s> "

                    chain = chain[:-10] + " "

                    encoding = self.tokenizer.encode_plus(
                        chain,
                        text_post,
                        max_length=self.max_len_bert,
                        add_special_tokens=True,
                        padding='max_length',
                        return_attention_mask=True,
                        return_token_type_ids=False,
                        return_tensors='pt',
                        truncation=False
                    )

                    if encoding['input_ids'].flatten().shape[0] <= self.max_len_bert:
                        flag = False
                    else:
                        # extreme case: if the only remained elements are the first and last one, truncate
                        if i >= len(dial) - 3:
                            chain = "<o> " + generate_descriptive_number_from_int(list_user.index(orig[-2])) + " user </o> " + \
                                    preprocess_text(dial[-2])

                            encoding = self.tokenizer.encode_plus(
                                chain,
                                text_post,
                                max_length=self.max_len_bert,
                                add_special_tokens=True,
                                padding='max_length',
                                return_attention_mask=True,
                                return_token_type_ids=False,
                                return_tensors='pt',
                                truncation='only_first'
                            )
                            flag = False
                        else:
                            i = i + 1

            # save the encoding
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
DataLoader for TC + U model, half training set per epoch
'''
def create_data_loaderData_TC_U(data, tokenizer, max_len_bert, batch_size, eval=False):
    ds = Data_TC_U(
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
Class to load the data for the TC + u + T model
<s> contextual_initial_claim </s></s> ... </s></s> contextual_target_claim-1 </s></s> contextual_target_claim </s>
where a contextual post is a post augmented with metadata (e.g. author, time, etc.) in this way:
contextual_post = <t> time </t> <o> author_id </o> post 

time is described in the following way: "after a days, b hours and c minutes"
author_id and parent_author_id are described in the following way: "x^th user"

The ids of authors are unique inside each dialogue
'''
class Data_TC_U_T(Dataset):

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
            # get dialogue, author sequence, parent sequence and time sequence
            dial = data['dialogue'][idx]
            orig = data['author_id'][idx]
            time = data['time'][idx]
            # save users in the dialogue in a list. The index is the local ID of the user
            list_user = list()

            chain = ""

            for idx_d in range(len(dial)-1):
                if orig[idx_d] not in list_user:
                    list_user.append(orig[idx_d])
                # concatenate iteratively contextual posts in chronological order
                chain = chain + "<t> " + generate_descriptive_date_from_seconds(
                    time[idx_d]) + " </t> <o> " + generate_descriptive_number_from_int(
                    list_user.index(orig[idx_d])) + " user </o> " + preprocess_text(dial[idx_d]) + " </s></s> "

            # remove last </s></s>
            chain = chain[:-10] + " "

            if orig[-1] not in list_user:
                list_user.append(orig[-1])
            text_post = "<t> " + generate_descriptive_date_from_seconds(time[-1]) + " </t> <o> " + \
                        generate_descriptive_number_from_int(list_user.index(orig[-1])) + " user </o> " + \
                        preprocess_text(dial[-1])

            encoding = self.tokenizer.encode_plus(
                chain,
                text_post,
                max_length=self.max_len_bert,
                add_special_tokens=True,
                padding='max_length',
                return_attention_mask=True,
                return_token_type_ids=False,
                return_tensors='pt',
                truncation=False
            )
            # if the input is too long, remove the iteratively the second post until the input is not too long (keep the root)
            if encoding['input_ids'].flatten().shape[0] > self.max_len_bert:

                flag = True
                i = 1

                while (flag):
                    chain = ""
                    for idx_d in range(len(dial)-1):
                        if idx_d == 0 or idx_d == len(dial)-2 or idx_d > i:
                            chain = chain + "<t> " + generate_descriptive_date_from_seconds(
                                time[idx_d]) + " </t> <o> " + generate_descriptive_number_from_int(
                                list_user.index(orig[idx_d])) + " user </o> " + preprocess_text(
                                dial[idx_d]) + " </s></s> "

                    chain = chain[:-10] + " "

                    encoding = self.tokenizer.encode_plus(
                        chain,
                        text_post,
                        max_length=self.max_len_bert,
                        add_special_tokens=True,
                        padding='max_length',
                        return_attention_mask=True,
                        return_token_type_ids=False,
                        return_tensors='pt',
                        truncation=False
                    )

                    if encoding['input_ids'].flatten().shape[0] <= self.max_len_bert:
                        flag = False
                    else:
                        # extreme case: if the only remained elements are the first and last one, truncate
                        if i >= len(dial) - 3:
                            chain = "<t> " + generate_descriptive_date_from_seconds(time[-2]) + " </t> <o> " + \
                            generate_descriptive_number_from_int(list_user.index(orig[-2])) + " user </o> " + \
                            preprocess_text(dial[-2])

                            encoding = self.tokenizer.encode_plus(
                                chain,
                                text_post,
                                max_length=self.max_len_bert,
                                add_special_tokens=True,
                                padding='max_length',
                                return_attention_mask=True,
                                return_token_type_ids=False,
                                return_tensors='pt',
                                truncation='only_first'
                            )
                            flag = False
                        else:
                            i = i + 1

            # save the encoding
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
DataLoader for TC + U + T model, half training set per epoch
'''
def create_data_loaderData_TC_U_T(data, tokenizer, max_len_bert, batch_size, eval=False):
    ds = Data_TC_U_T(
        data=data,
        tokenizer=tokenizer,
        max_len_bert=max_len_bert
    )
    if not eval:
        return DataLoader(ds, batch_size=batch_size, num_workers=4,
                          sampler=RandomSampler(ds, num_samples=len(ds)//2, replacement=False, generator=None)), len(ds)//2

    else:
        return DataLoader(ds, batch_size=batch_size, num_workers=4, shuffle=False), len(ds)
