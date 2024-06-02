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
    mlb = MultiLabelBinarizer()

    # Transform ground_truth and predicted data into binary arrays
    ground_truth_labels = mlb.fit_transform(ground_truth)
    pred_labels = mlb.transform(predicted)

    tp = 0          # T
    tn = 0
    fp = 0          # O 
    fn = 0

    for gt, pred in zip(ground_truth, predicted):
        if gt == 'T' and gt == pred:
            tp += 1
        elif gt == 'O' and gt == pred:
            tn += 1
        elif gt == 'T' and gt != pred:
            fn += 1
        elif gt == 'O' and gt != pred:
            fp += 1
    
    prec = tp/(tp + fp + SMALL_POSITIVE_CONST)
    rec = tp/(tp + fn + SMALL_POSITIVE_CONST)
    f1_ = 2 * (prec * rec) / (prec + rec + SMALL_POSITIVE_CONST)

    print("Precision: ", prec, ", Recall: ", rec, ", F1: ", f1_)


    # print("\n\n----------------> DATA: ", predicted[0], "\n (len: ", len(predicted[0]), ")\n")
    # print("----------------> LABELS:", pred_labels[0],"\n (len: ", len(pred_labels[0]), ")\n")

    # print("-"*89)

    # print("\n\n----------------> DATA: ", ground_truth[0], "\n (len: ", len(ground_truth[0]), ")\n")
    # print("----------------> LABELS:", ground_truth_labels[0],"\n (len: ", len(ground_truth_labels[0]), ")\n")

    precision, recall, f1, _ = precision_recall_fscore_support(ground_truth_labels, pred_labels, average='macro')

    print("Precision: ", precision, ", Recall: ", recall, ", F1: ", f1)
    return {'precision': precision, 'recall': recall, 'f1': f1}
