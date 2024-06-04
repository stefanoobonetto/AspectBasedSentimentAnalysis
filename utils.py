# Add functions or classes used for data loading and preprocessing

import torch
import json
import re
import torch.utils.data as data

PAD_TOKEN = 0

from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased") # Download the tokenizer

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_data(path):
    '''
        input: path/to/data
        output: json
    '''
    dataset = []
    with open(path) as f:
        dataset = json.loads(f.read())
    return dataset


class Lang():
    def __init__(self, slots, cutoff=0):
        self.slot2id = self.lab2id(slots)
        self.id2slot = {v:k for k, v in self.slot2id.items()}

    # creates a dictionary that maps unique id to each slot in the vocab

    def lab2id(self, elements, pad=True):                               
        vocab = {}
        if pad:
            vocab['pad'] = PAD_TOKEN
        for elem in elements:
                vocab[elem] = len(vocab)
        return vocab

class Slots (data.Dataset):
    # Mandatory methods are __init__, __len__ and __getitem__

    def __init__(self, dataset, lang, unk='unk'):
        self.utterances = []
        self.attention_mask = []
        self.token_type_ids = []

        self.slots = []

        self.slot_ids = []
        self.utt_ids = []

        self.unk = unk

        for x in dataset:
            self.utterances.append("[CLS] " + x['utterance'] + " [SEP]")                # add typical BOF adn EOF BERT tags 
            self.slots.append("O " + x['slots'] + " O")                                 # and their respective slots 

        self.utt_ids, self.slot_ids, self.attention_mask, self.token_type_ids = self.mapping_seq(self.utterances, self.slots, tokenizer, lang.slot2id)

    def __len__(self):
        return len(self.utterances)

    def __getitem__(self, idx):
        utt = torch.Tensor(self.utt_ids[idx])
        att = torch.Tensor(self.attention_mask[idx])
        token_type = torch.Tensor(self.token_type_ids[idx])
        slots = torch.Tensor(self.slot_ids[idx])
        sample = {'utterance': utt, 'attention_mask': att, 'token_type_ids': token_type, 'slots': slots}
        return sample

    def mapping_seq(self, utterances, slots, tokenizer, mapper_slot):  # map sequences to number
        res_utt = []  
        res_slot = []  
        res_attention = []  
        res_token_type_id = []

        for sequence, slot in zip(utterances, slots):  # iterate through each sequence and its corresponding slots
            tmp_seq = []  
            tmp_slot = [] 
            tmp_attention = []  
            tmp_token_type_id = []  

            for word, elem in zip(sequence.split(), slot.split(' ')):  # iterate through each word and its corresponding slot tag
                tmp_attention.append(1)                         # append 1 to attention mask for the word
                tmp_token_type_id.append(0)                     # append 0 to token type ids for the word
                word_tokens = tokenizer(word)                   # tokenize the word with BERT
                word_tokens = word_tokens[1:-1]                 # remove special tokens ([CLS] and [SEP])

                tmp_seq.extend(word_tokens['input_ids'])  
                tmp_slot.extend([mapper_slot[elem]] + [mapper_slot['pad']] * (len(word_tokens['input_ids']) - 1))  # add slot tag and pad for sub-tokens

                for i in range(len(word_tokens['input_ids']) - 1):  
                    tmp_attention.append(0)                         # append 0 to attention mask for sub-tokens
                    tmp_token_type_id.append(0)  

            res_utt.append(tmp_seq)  
            res_slot.append(tmp_slot)  
            res_attention.append(tmp_attention) 
            res_token_type_id.append(tmp_token_type_id)

        return res_utt, res_slot, res_attention, res_token_type_id  

def collate_fn(data):
    def merge(sequences):
        '''
        merge from batch * sent_len to batch * max_len
        '''
        lengths = [len(seq) for seq in sequences]
        max_len = 1 if max(lengths)==0 else max(lengths)

        # Pad token is zero in our case
        # So we create a matrix full of PAD_TOKEN (i.e. 0) with the shape
        # batch_size X maximum length of a sequence

        padded_seqs = torch.LongTensor(len(sequences),max_len).fill_(PAD_TOKEN)
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq                  # copy each sequence into the matrix, substituting 0 with respective ids of the words (if present)
        padded_seqs = padded_seqs.detach()  
        return padded_seqs, lengths

    data.sort(key=lambda x: len(x['utterance']), reverse=True)
    new_item = {}
    for key in data[0].keys():
        new_item[key] = [d[key] for d in data]

    # We just need one length for packed pad seq, since len(utt) == len(slots)
    src_utt, _ = merge(new_item['utterance'])                  
    attention, _ = merge(new_item['attention_mask'])                  
    token_type_ids, _ = merge(new_item['token_type_ids'])                  

    y_slots, y_lengths = merge(new_item["slots"])

    src_utt = src_utt.to(device) 
    y_slots = y_slots.to(device)
    attention = attention.to(device)
    token_type_ids = token_type_ids.to(device)
    y_lengths = torch.LongTensor(y_lengths).to(device)

    new_item["utterances"] = src_utt
    new_item["attention_mask"] = attention
    new_item["token_type_ids"] = token_type_ids
    new_item["y_slots"] = y_slots
    new_item["slots_len"] = y_lengths

    return new_item

def convert_text_to_json(input_file, output_file):
    
    # this function picks up utterances from the right of #### in the txt dataset (utt = slot) 

    data = []

    with open(input_file, 'r') as file:
        for line in file:
            parts = line.strip().split('####')
            if len(parts) == 2:
                tags_part = parts[1].strip()
                words_and_tags = tags_part.split()

                utterance = []
                slots = []

                for word_tag in words_and_tags:
                    
                    match = re.match(r'(.+)=([^-]+(?:-[^-]+)*)', word_tag)                  # match the word and the tag using a regular expression
                    if match:
                        word, tag = match.groups()
                        utterance.append(word)
                        slots.append('O' if tag.startswith('O') else 'T')                   # don't need POS and NEG

                utterance_str = ' '.join(utterance)
                slots_str = ' '.join(slots)

                entry = {
                    "utterance": utterance_str,
                    "slots": slots_str
                }

                data.append(entry)

    with open(output_file, 'w') as outfile:
        json.dump(data, outfile, indent=4)

    print("Correctly convert ", input_file, " into ", output_file)


