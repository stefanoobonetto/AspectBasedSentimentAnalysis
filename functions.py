# Add the class of your model only
# Here is where you define the architecture of your model using pytorch

import torch.nn as nn
import torch
from sklearn.metrics import precision_recall_fscore_support

from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased") # Download the tokenizer

SMALL_POSITIVE_CONST = 1e-6

def train_loop(data, optimizer, criterion_slots, model, clip=5):
    model.train()
    loss_array = []
    for sample in data:
        optimizer.zero_grad() # Zeroing the gradient
        # print(sample['utterances'], sample['attention_mask'], sample['token_type_ids'])
        slots = model(sample['utterances'], attentions=sample['attention_mask'], token_type_ids=sample['token_type_ids'])

        # print("shape gt: ", sample['y_slots'].shape, ", shape pred: ", slots.shape)

        loss = criterion_slots(slots, sample['y_slots'])

        loss_array.append(loss.item())
        loss.backward()  
                        
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)  
        optimizer.step() # Update the weights
    return loss_array


def eval_loop(data, criterion_slots, model, lang):
    model.eval()
    loss_array = []
    
    ref_slots = []
    hyp_slots = []
    #softmax = nn.Softmax(dim=1) # Use Softmax if you need the actual probability
    with torch.no_grad(): # It used to avoid the creation of computational graph
        for sample in data:
            slots = model(sample['utterances'], attentions=sample['attention_mask'], token_type_ids=sample['token_type_ids'])
            
            loss = criterion_slots(slots, sample['y_slots'])
            
            loss_array.append(loss.item())

            # Slot inference 
            output_slots = torch.argmax(slots, dim=1)

            for id_seq, seq in enumerate(output_slots):
                
                length = sample['slots_len'].tolist()[id_seq]
                utt_ids = sample['utterance'][id_seq][:length].tolist()
                gt_ids = sample['y_slots'][id_seq].tolist()
                gt_slots = [lang.id2slot[elem] for elem in gt_ids[:length]]           
                utterance = tokenizer.convert_ids_to_tokens(utt_ids)
                to_decode = seq[:length].tolist()
                ref_slots.append([(utterance[id_el], elem) for id_el, elem in enumerate(gt_slots)])
                tmp_seq = []
                for id_el, elem in enumerate(to_decode):
                    tmp_seq.append((utterance[id_el], lang.id2slot[elem]))
                hyp_slots.append(tmp_seq)
                # print(to_decode)


        tmp_ref = []
        tmp_hyp = []
        tmp_ref_tot = []
        tmp_hyp_tot = []

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

        # print("results: ", results)
        
    except Exception as ex:
        # Sometimes the model predicts a class that is not in REF
        print("Warning:", ex)


        ref_s = set([x[1] for x in ref_slots])
        hyp_s = set([x[1] for x in hyp_slots])
        print(hyp_s.difference(ref_s))
        results =  {'precision' : 0, 'recall' : 0, 'f1' : 0} 
    # for elem in report_intent:
        # print(elem, report_intent[elem], "\n\n")
    
    # print("report_intent: ", report_intent)
    return results, loss_array


def evaluate(gold_data, pred_data):
    assert len(gold_data) == len(pred_data)
    n_sent = len(gold_data)

    all_gold_slots = []
    all_pred_slots = []

    for i in range(n_sent):
        print("gold_data: ", gold_data[i])
        for j, gold in enumerate(gold_data[i]):
            pred = pred_data[i]
            utterance = gold[0]
            print("\nutterance: ", utterance)
            gold_slots = gold[1]
            print(gold_slots)
            pred_slots = pred[j][1]

            for id, tag, utt, in zip(enumerate(gold_slots), utterance):
                if tag != 'pad' and utt != 'CLS' and utt != '[SEP]':
                    all_gold_slots.append(tag)
                    all_pred_slots.append(pred_slots[id])
        print("\n\n")
        # Filter out the 'X' labels for evaluation
        # filtered_gold_slots = [tag for tag, utt in zip(gold_slots, utterance) if tag != 'X' and utt != '[CLS]' and utt != '[S]']
        # filtered_pred_slots = [tag for tag in pred_slots if tag != 'X']

        # all_gold_slots.extend(filtered_gold_slots)
        # all_pred_slots.extend(filtered_pred_slots)

    # Calculate precision, recall, and f1
    precision, recall, f1, _ = precision_recall_fscore_support(all_gold_slots, all_pred_slots, average='macro')
    return {'precision' : precision, 'recall' : recall, 'f1' : f1}


# def evaluate(gold_ot, pred_ot):
#     """
#     evaluate the model performce for the ote task
#     :param gold_ot: gold standard ote tags
#     :param pred_ot: predicted ote tags
#     :return:
#     """
#     assert len(gold_ot) == len(pred_ot)
#     n_samples = len(gold_ot)
#     # number of true positive, gold standard, predicted opinion targets
#     n_tp_ot, n_gold_ot, n_pred_ot = 0, 0, 0
#     for i in range(n_samples):
#         g_ot = gold_ot[i]
#         p_ot = pred_ot[i]
#         g_ot_sequence, p_ot_sequence = tag2ot(ote_tag_sequence=g_ot), tag2ot(ote_tag_sequence=p_ot)
#         # hit number
#         n_hit_ot = match_ot(gold_ote_sequence=g_ot_sequence, pred_ote_sequence=p_ot_sequence)
#         n_tp_ot += n_hit_ot
#         n_gold_ot += len(g_ot_sequence)
#         n_pred_ot += len(p_ot_sequence)
#     # add 0.001 for smoothing
#     # calculate precision, recall and f1 for ote task
#     ot_precision = float(n_tp_ot) / float(n_pred_ot + SMALL_POSITIVE_CONST)
#     ot_recall = float(n_tp_ot) / float(n_gold_ot + SMALL_POSITIVE_CONST)
#     ot_f1 = 2 * ot_precision * ot_recall / (ot_precision + ot_recall + SMALL_POSITIVE_CONST)
#     ote_scores = (ot_precision, ot_recall, ot_f1)
#     return ote_scores


# def match_ot(gold_ote_sequence, pred_ote_sequence):
#     """
#     calculate the number of correctly predicted opinion target
#     :param gold_ote_sequence: gold standard opinion target sequence
#     :param pred_ote_sequence: predicted opinion target sequence
#     :return: matched number
#     """
#     n_hit = 0
#     for t in pred_ote_sequence:
#         if t in gold_ote_sequence:
#             n_hit += 1
#     return n_hit

