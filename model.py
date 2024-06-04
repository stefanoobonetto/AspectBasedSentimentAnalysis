import torch.nn as nn
from transformers.models.bert.modeling_bert import BertModel

class JointBERT(nn.Module):

    def __init__(self, hid_size, out_slot):
        super(JointBERT, self).__init__()

        self.bert = BertModel.from_pretrained('bert-base-uncased')
    
        self.slot_out = nn.Linear(hid_size, out_slot)

    def forward(self, utterances, attentions=None, token_type_ids=None):
        
        # Get the BERT output
        outputs = self.bert(utterances, attention_mask=attentions, token_type_ids=token_type_ids)

        sequence_output = outputs[0]
        
        slots = self.slot_out(sequence_output)
        
        slots = slots.permute(0,2,1) 

        return slots
    


