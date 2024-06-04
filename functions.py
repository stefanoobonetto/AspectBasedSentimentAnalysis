import torch
from torch.nn.utils import clip_grad_norm_

from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased") # Download the tokenizer

SMALL_POSITIVE_CONST = 1e-6

def train_loop(data, optimizer, criterion_slots, model, clip=5):
    model.train()
    loss_array = []
    for sample in data:
        optimizer.zero_grad() 

        slots = model(sample['utterances'], attentions=sample['attention_mask'], token_type_ids=sample['token_type_ids'])

        loss = criterion_slots(slots, sample['y_slots'])

        loss_array.append(loss.item())
        loss.backward()  
                        
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)  
        optimizer.step() 
    return loss_array


def eval_loop(data, criterion_slots, model, lang):
    
    model.eval()
    loss_array = []
    
    ref_slots = []
    hyp_slots = []

    with torch.no_grad(): 

        for sample in data:
        
            slots = model(sample['utterances'], attentions=sample['attention_mask'], token_type_ids=sample['token_type_ids'])
            
            loss = criterion_slots(slots, sample['y_slots'])
            
            loss_array.append(loss.item())

            # Slot inference 
            output_slots = torch.argmax(slots, dim=1)

            for id_seq, seq in enumerate(output_slots):
                
                lenground_truthh = sample['slots_len'].tolist()[id_seq]
                utt_ids = sample['utterance'][id_seq][:lenground_truthh].tolist()
                ground_truth_ids = sample['y_slots'][id_seq].tolist()
                ground_truth_slots = [lang.id2slot[elem] for elem in ground_truth_ids[:lenground_truthh]]           
                utterance = tokenizer.convert_ids_to_tokens(utt_ids)
                to_decode = seq[:lenground_truthh].tolist()
                ref_slots.append([(utterance[id_el], elem) for id_el, elem in enumerate(ground_truth_slots)])
                tmp_seq = []

                for id_el, elem in enumerate(to_decode):
                    tmp_seq.append((utterance[id_el], lang.id2slot[elem]))
                hyp_slots.append(tmp_seq)

        tmp_ref = []
        tmp_hyp = []
        tmp_ref_tot = []
        tmp_hyp_tot = []

        # this part is done to remove pad in the reference slots and in the relative position in the predicted slot  

        for ref, hyp in zip(ref_slots, hyp_slots):
            tmp_ref = []
            tmp_hyp = []

            for r, h in zip(ref, hyp):
                if r[1] != 'pad' and r[0] != '[CLS]' and r[0] != '[SEP]':
                    tmp_ref.append(r)
                    tmp_hyp.append(h)
            
            tmp_ref_tot.append(tmp_ref)
            tmp_hyp_tot.append(tmp_hyp)
        
        ref_slots = tmp_ref_tot
        hyp_slots = tmp_hyp_tot

    try:            
        
        results = evaluate(tmp_ref_tot, tmp_hyp_tot)

    except Exception as ex:

        print("Warning:", ex)

        ref_s = set([x[1] for x in ref_slots])
        hyp_s = set([x[1] for x in hyp_slots])
        print(hyp_s.difference(ref_s))
        results = {"f1" :0}

    return results, loss_array

def evaluate(ground_truth, predicted):
    
    tp = 0          # 'T' is considered to be the positive class
    fp = 0           
    fn = 0

    # for each slot predicted of the sentence I compare the prediction with the reference and I calculate TN, FN and FP

    for gt_sent, pred_sent in zip(ground_truth, predicted):        
        for gt, pred in zip(gt_sent, pred_sent):

            if gt[1] == 'T' and gt[1] == pred[1]:
                tp += 1
            elif gt[1] == 'T' and gt[1] != pred[1]:
                fn += 1
            elif gt[1] == 'O' and gt[1] != pred[1]:
                fp += 1
    
    # print("TP: ", tp, "FP: ", fp, "FN: ", fn)

    # the SMALL_POSITIVE_CONST is used to avoid /0

    precision = tp/(tp + fp + SMALL_POSITIVE_CONST)
    recall = tp/(tp + fn + SMALL_POSITIVE_CONST)
    f1 = 2 * (precision * recall) / (precision + recall + SMALL_POSITIVE_CONST)


    # uncomment below to see epoch by epoch the statistics

    # print("Precision: ", precision, ", Recall: ", recall, ", F1: ", f1)

    return {"precision" : precision, "recall" : recall, "f1" : f1}
