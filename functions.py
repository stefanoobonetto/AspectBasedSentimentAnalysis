import torch.nn as nn
import torch
from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import MultiLabelBinarizer

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

        # print("shape ground_truth: ", sample['y_slots'].shape, ", shape pred: ", slots.shape)

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

    with torch.no_grad(): # It used to avoid the creation of computational graph
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

    # for elem in report_intent:
        # print(elem, report_intent[elem], "\n\n")
    
    # print("report_intent: ", report_intent)
    return results, loss_array

def evaluate(ground_truth, predicted):
    
    tp = 0          # 'T' is considered to be the positive class
    fp = 0           
    fn = 0

    for gt_sent, pred_sent in zip(ground_truth, predicted):
        
        for gt, pred in zip(gt_sent, pred_sent):

            if gt[1] == 'T' and gt[1] == pred[1]:
                tp += 1
            elif gt[1] == 'T' and gt[1] != pred[1]:
                fn += 1
            elif gt[1] == 'O' and gt[1] != pred[1]:
                fp += 1
    
    print("TP: ", tp, "FP: ", fp, "FN: ", fn)

    prec = tp/(tp + fp + SMALL_POSITIVE_CONST)
    rec = tp/(tp + fn + SMALL_POSITIVE_CONST)
    # f1_ = 2 * (prec * rec) / (prec + rec + SMALL_POSITIVE_CONST)
    f1_ = tp/(tp + (fn + fp)/2 + SMALL_POSITIVE_CONST)

    print("Precision: ", prec, ", Recall: ", rec, ", F1: ", f1_)

    mlb = MultiLabelBinarizer()

    ground_truth_labels = mlb.fit_transform(ground_truth)
    pred_labels = mlb.transform(predicted)

    precision, recall, f1, _ = precision_recall_fscore_support(ground_truth_labels, pred_labels, average='macro')

    print("[sklearn] Precision: ", precision, ", Recall: ", recall, ", F1: ", f1)

    return {"precision" : prec, "recall" : rec, "f1" : f1_}

def evaluate_(ground_truth, predicted):

    # Transform ground_truth and predicted data into binary arrays
    return {'precision': precision, 'recall': recall, 'f1': f1}
