# Add functions or classes used for data loading and preprocessing

import torch
import json
import re
from collections import Counter
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

    def lab2id(self, elements, pad=True):                               # creates a dictionary that maps unique id to each label
        vocab = {}
        if pad:
            vocab['pad'] = PAD_TOKEN
        for elem in elements:
                vocab[elem] = len(vocab)
        # vocab['unk'] = len(vocab)
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
            self.utterances.append("[CLS] " + x['utterance'] + " [SEP]")
            self.slots.append("O " + x['slots'] + " O")

        self.utt_ids, self.slot_ids, self.attention_mask, self.token_type_ids = self.mapping_seq(self.utterances, self.slots, tokenizer, lang.slot2id)


        # for intent, intent_id in zip(self.intents, self.intent_ids):
        #     print("Intent: ", str(intent))
        #     print("Intent[IDs]: ", str(intent_id))
        #     print("Translated id to token: ", lang.id2intent[intent_id])

        # for i in range(len(self.utterances)):
        #     print("Phrase:                      ", self.utterances[i])
        #     print("Phrase[IDs]:                 ", self.utt_ids[i]['input_ids'])

        #     # print("last_id: ", self.utt_ids[i]['input_ids'][-1], " correspond to ", tokenizer.convert_ids_to_tokens([self.utt_ids[i]['input_ids'][-1]]))
        #     print("Slots:                       ", self.slots[i])
        #     print("Slots[IDs]:                  ", self.slot_ids[i]['input_ids'])
        #     print("Intent[IDs]:                 ", self.intent_ids[i]['input_ids'])

        #     print("\n\n")

    def __len__(self):
        return len(self.utterances)

    def __getitem__(self, idx):
        utt = torch.Tensor(self.utt_ids[idx])
        att = torch.Tensor(self.attention_mask[idx])
        token_type = torch.Tensor(self.token_type_ids[idx])
        slots = torch.Tensor(self.slot_ids[idx])
        sample = {'utterance': utt, 'attention_mask': att, 'token_type_ids': token_type, 'slots': slots}
        return sample

    def mapping_seq(self, utterances, slots, tokenizer, mapper_slot): # Map sequences to number
        res_utt = []
        res_slot = []
        res_attention = []
        res_token_type_id = []

        for sequence, slot in zip(utterances, slots):
            # print("utterance: ", sequence, ",            len: ", len(sequence.split()), "len_bert: ", len(tokenizer(sequence)['input_ids']))
            tmp_seq = []
            tmp_slot = []
            tmp_attention = []
            tmp_token_type_id = []

            for word, elem in zip(sequence.split(), slot.split(' ')):
                tmp_attention.append(1)
                tmp_token_type_id.append(0)
                word_tokens = tokenizer(word)
                word_tokens = word_tokens[1:-1]

                tmp_seq.extend(word_tokens['input_ids'])
                tmp_slot.extend([mapper_slot[elem]] + [mapper_slot['pad']] * (len(word_tokens['input_ids']) - 1))

                for i in range(len(word_tokens['input_ids']) - 1):
                    tmp_attention.append(0)
                    tmp_token_type_id.append(0)

            res_utt.append(tmp_seq)
            res_slot.append(tmp_slot)
            res_attention.append(tmp_attention)
            res_token_type_id.append(tmp_token_type_id)

            # print("utterance: ", tokenizer.tokenize(sequence), ",            len: ", len(tokenizer.tokenize(sequence)), "\nutterance_bert: ", tokenizer.convert_ids_to_tokens(tmp_seq) ,"len_bert: ", len(tmp_seq), "\n\n")

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
            padded_seqs[i, :end] = seq # We copy each sequence into the matrix
        # print(padded_seqs)
        padded_seqs = padded_seqs.detach()  # We remove these tensors from the computational graph
        return padded_seqs, lengths
    # Sort data by seq lengths


    data.sort(key=lambda x: len(x['utterance']), reverse=True)
    new_item = {}
    for key in data[0].keys():
        new_item[key] = [d[key] for d in data]

    # We just need one length for packed pad seq, since len(utt) == len(slots)
    src_utt, _ = merge(new_item['utterance'])                  # input_ids': utt, 'attention_mask': att, 'token_type_ids': token})
    attention, _ = merge(new_item['attention_mask'])                  # input_ids': utt, 'attention_mask': att, 'token_type_ids': token})
    token_type_ids, _ = merge(new_item['token_type_ids'])                  # input_ids': utt, 'attention_mask': att, 'token_type_ids': token})

    y_slots, y_lengths = merge(new_item["slots"])

    # print("y_slots: ", y_slots)
    # print("intent: ", new_item["intent"])
    # intent = pad_sequence(intent, batch_first=True, padding_value=0)        # Pad the sequences (some intents may be composed by more tokens)

    src_utt = src_utt.to(device) # We load the Tensor on our selected device
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



def convert_text_to_json_(input_file, output_file):

    # this function picks up utterances from the left of #### in the txt dataset

    data = []

    with open(input_file, 'r') as file:
        for line in file:
            parts = line.strip().split('####')
            utterance = ""
            slots = ""
            if len(parts) == 2:             
                utterance = parts[0].strip()
                tags = parts[1].strip().split()
                new_tags = []
                for tag in tags:
                    tag = tag.split('=')[1] if len(tag.split('=')) < 3 else tag.split('=')[2]
                    new_tags.append(tag)
                tags = new_tags

                slots = ' '.join([tag if tag.startswith('O') else 'T' for tag in tags])

                entry = {
                    "utterance": utterance,
                    "slots": slots
                }

                data.append(entry)

    with open(output_file, 'w') as outfile:
        json.dump(data, outfile, indent=4)

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
                    # Match the word and the tag using a regular expression
                    match = re.match(r'(.+)=([^-]+(?:-[^-]+)*)', word_tag)
                    if match:
                        word, tag = match.groups()
                        utterance.append(word)
                        slots.append('O' if tag.startswith('O') else 'T')

                # Join the lists into strings
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




# for elem in parts[1].split(" "):
#                     utterance += elem.split("=")[0]
#                     utterance += " "
#                     slots += 'O' if elem.split("=")[len(elem.split("="))-1].startswith('O') else 'T'
#                     slots += " "