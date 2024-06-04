import torch.nn as nn
from transformers.models.bert.modeling_bert import BertModel

class JointBERT(nn.Module):

    def __init__(self, hid_size, out_slot):
        super(JointBERT, self).__init__()

        self.bert = BertModel.from_pretrained('bert-base-uncased')
    
        self.slot_out = nn.Linear(hid_size, out_slot)

    def forward(self, utterances, attentions=None, token_type_ids=None):
        
        # get the BERT output
        outputs = self.bert(utterances, attention_mask=attentions, token_type_ids=token_type_ids)

        sequence_output = outputs[0]  # extract the sequence output from BERT corresponding to the slots prediction

        slots = self.slot_out(sequence_output)  # pass it through the slot output layer

        # Slot size: batch_size, seq_len, classes 
        slots = slots.permute(0, 2, 1)  # permute the slots tensor to match the expected shape
        # Slot size: batch_size, classes, seq_len
        return slots  
        


