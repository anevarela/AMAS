# train.py
import argparse
import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, AutoConfig, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score
import ast

MODEL_MAPPING = {
    "rigoberta_clinical": "IIC/RigoBERTa-Clinical",
    "bsc_bio": "PlanTL-GOB-ES/bsc-bio-es",
    "roberta_biomedical": "PlanTL-GOB-ES/roberta-base-biomedical-es",
    "gatortron": "UFNLP/gatortron-base",
    "bert_spanish": "dccuchile/bert-base-spanish-wwm-cased",
    "mbert": "google-bert/bert-base-multilingual-cased",
    "rigoberta_general": "IIC/RigoBERTa-2.0",
    "eriberta": "HiTZ/EriBERTa-base"
}

CLASSIFICATION_COLS = [
    'pred_sdoh_community_present', 'pred_sdoh_community_absent', 'pred_sdoh_education',
    'pred_sdoh_economics', 'pred_sdoh_environment', 'pred_behavior_alcohol',
    'pred_behavior_tobacco', 'pred_behavior_drug'
]

class ClassificationDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=512):
        self.texts = df['text_sp'].fillna("").astype(str).tolist()
        # Parse targets as integer class indices for CrossEntropy compatibility
        self.labels = df[CLASSIFICATION_COLS].values.astype(np.int64)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx], truncation=True, padding='max_length',
            max_length=self.max_len, return_tensors="pt"
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

class MultiModifierNERDataset(Dataset):
    def __init__(self, df, tokenizer, tag2idx, max_len=512):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.tag2idx = tag2idx
        self.data = []
        
        for _, row in df.iterrows():
            gold_ner = row['gold_ner']
            if isinstance(gold_ner, str):
                try:
                    data = json.loads(gold_ner)
                except json.JSONDecodeError:
                    try:
                        data = ast.literal_eval(gold_ner)
                    except:
                        continue
            else:
                data = gold_ner
            
            words = [item[0] for item in data]
            raw_tags = [item[1] for item in data]
            self.data.append((words, raw_tags))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        words, raw_tags = self.data[idx]
        encoding = self.tokenizer(
            words, is_split_into_words=True, truncation=True,
            padding='max_length', max_length=self.max_len, return_tensors="pt"
        )
        
        word_ids = encoding.word_ids()
        base_labels, neg_labels, quant_labels, temp_labels = [], [], [], []
        
        for word_idx in word_ids:
            if word_idx is None:
                base_labels.append(-100)
                neg_labels.append(-100)
                quant_labels.append(-100)
                temp_labels.append(-100)
            else:
                raw_tag = str(raw_tags[word_idx])
                
                # modifiers
                is_negated = 1 if "+NEGATION" in raw_tag else 0
                is_quantity = 1 if "QUANTITY" in raw_tag else 0
                is_temporality = 1 if "TEMPORALITY" in raw_tag else 0
                
                # clean base sdoh tag
                clean_tag = raw_tag.replace("+NEGATION", "").replace("QUANTITY", "").replace("TEMPORALITY", "").split("-")[-1].strip()
                if clean_tag.lower() == 'o' or clean_tag == "":
                    clean_tag = "O"
                
                base_labels.append(self.tag2idx.get(clean_tag, self.tag2idx["O"]))
                neg_labels.append(is_negated)
                quant_labels.append(is_quantity)
                temp_labels.append(is_temporality)
                
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'base_labels': torch.tensor(base_labels, dtype=torch.long),
            'neg_labels': torch.tensor(neg_labels, dtype=torch.long),
            'quant_labels': torch.tensor(quant_labels, dtype=torch.long),
            'temp_labels': torch.tensor(temp_labels, dtype=torch.long)
        }

class MultiHeadClassifier(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        
        # Dynamically define number of classes per target head
        self.head_classes = []
        for col in CLASSIFICATION_COLS:
            if 'community' in col or 'education' in col:
                self.head_classes.append(2)
            else:
                self.head_classes.append(3)
                
        self.heads = nn.ModuleList([
            nn.Linear(self.config.hidden_size, num_classes) for num_classes in self.head_classes
        ])
        
    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]
        return [head(pooled) for head in self.heads]

class MultiHeadTokenClassifier(nn.Module):
    def __init__(self, model_name, num_base_labels):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        
        # Shared core base model with independent linear token-classification heads
        self.base_classifier = nn.Linear(self.config.hidden_size, num_base_labels)
        self.neg_classifier = nn.Linear(self.config.hidden_size, 2)
        self.quant_classifier = nn.Linear(self.config.hidden_size, 2)
        self.temp_classifier = nn.Linear(self.config.hidden_size, 2)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        return {
            'base': self.base_classifier(sequence_output),
            'neg': self.neg_classifier(sequence_output),
            'quant': self.quant_classifier(sequence_output),
            'temp': self.temp_classifier(sequence_output)
        }

def train_epoch(models, loaders, optimizers, schedulers, device, task):
    criterion_ce = nn.CrossEntropyLoss(ignore_index=-100)
    criterion_cls_ce = nn.CrossEntropyLoss()
    total_loss = 0
    
    if task == 'classification':
        models['cls'].train()
        for batch in loaders:
            optimizers['cls'].zero_grad()
            ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            logits_list = models['cls'](ids, mask)
            loss = 0
            for i, logits in enumerate(logits_list):
                loss += criterion_cls_ce(logits, labels[:, i])
                
            loss.backward()
            optimizers['cls'].step()
            schedulers['cls'].step()
            total_loss += loss.item()
    else:
        models['ner'].train()
        optimizers['ner'].zero_grad()
            
        for batch in loaders:
            ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            
            l_base = batch['base_labels'].to(device)
            l_neg = batch['neg_labels'].to(device)
            l_quant = batch['quant_labels'].to(device)
            l_temp = batch['temp_labels'].to(device)
            
            out = models['ner'](ids, mask)
            
            loss_base = criterion_ce(out['base'].view(-1, out['base'].shape[-1]), l_base.view(-1))
            loss_neg = criterion_ce(out['neg'].view(-1, out['neg'].shape[-1]), l_neg.view(-1))
            loss_quant = criterion_ce(out['quant'].view(-1, out['quant'].shape[-1]), l_quant.view(-1))
            loss_temp = criterion_ce(out['temp'].view(-1, out['temp'].shape[-1]), l_temp.view(-1))
            
            loss = loss_base + loss_neg + loss_quant + loss_temp
            loss.backward()
            
            optimizers['ner'].step()
            schedulers['ner'].step()
            optimizers['ner'].zero_grad()
                
            total_loss += loss.item()
            
    return total_loss / len(loaders)

def eval_model(models, loader, device, task):
    if task == 'classification':
        models['cls'].eval()
    else:
        models['ner'].eval()
        
    preds_dict = {'base': [], 'neg': [], 'quant': [], 'temp': [], 'cls': []}
    labels_dict = {'base': [], 'neg': [], 'quant': [], 'temp': [], 'cls': []}
    
    with torch.no_grad():
        for batch in loader:
            ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            
            if task == 'classification':
                labels = batch['labels'].numpy()
                logits_list = models['cls'](ids, mask)
                
                # Combine predictions across distinct classification heads
                preds = np.stack([torch.argmax(logits, dim=-1).cpu().numpy() for logits in logits_list], axis=1)
                preds_dict['cls'].append(preds)
                labels_dict['cls'].append(labels)
            else:
                l_b, l_n = batch['base_labels'].numpy(), batch['neg_labels'].numpy()
                l_q, l_t = batch['quant_labels'].numpy(), batch['temp_labels'].numpy()
                
                out = models['ner'](ids, mask)
                o_b = torch.argmax(out['base'], dim=-1).cpu().numpy()
                o_n = torch.argmax(out['neg'], dim=-1).cpu().numpy()
                o_q = torch.argmax(out['quant'], dim=-1).cpu().numpy()
                o_t = torch.argmax(out['temp'], dim=-1).cpu().numpy()
                
                for pb, pn, pq, pt, lb, ln, lq, lt in zip(o_b, o_n, o_q, o_t, l_b, l_n, l_q, l_t):
                    v = lb != -100
                    preds_dict['base'].extend(pb[v]); labels_dict['base'].extend(lb[v])
                    preds_dict['neg'].extend(pn[v]); labels_dict['neg'].extend(ln[v])
                    preds_dict['quant'].extend(pq[v]); labels_dict['quant'].extend(lq[v])
                    preds_dict['temp'].extend(pt[v]); labels_dict['temp'].extend(lt[v])
                    
    if task == 'classification':
        all_labels_cls = np.concatenate(labels_dict['cls'], axis=0).astype(int)
        all_preds_cls = np.concatenate(preds_dict['cls'], axis=0).astype(int)
        
        num_heads = all_labels_cls.shape[1]
        f1_per_column = []
        
        for c in range(num_heads):
            col_f1 = f1_score(all_labels_cls[:, c], all_preds_cls[:, c], average='macro', zero_division=0)
            f1_per_column.append(col_f1)
            
        return np.mean(f1_per_column)
        
    else:
        f1_b = f1_score(labels_dict['base'], preds_dict['base'], average='macro', zero_division=0)
        f1_n = f1_score(labels_dict['neg'], preds_dict['neg'], average='macro', zero_division=0)
        f1_q = f1_score(labels_dict['quant'], preds_dict['quant'], average='macro', zero_division=0)
        f1_t = f1_score(labels_dict['temp'], preds_dict['temp'], average='macro', zero_division=0)
        return (f1_b + f1_n + f1_q + f1_t) / 4.0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', type=str, required=True)
    parser.add_argument('--dev_path', type=str, required=True)
    parser.add_argument('--model_key', type=str, required=True, choices=MODEL_MAPPING.keys())
    parser.add_argument('--task', type=str, required=True, choices=['classification', 'ner'])
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_name = MODEL_MAPPING[args.model_key]
    if "hitz" in model_name.lower():
        tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    train_df = pd.read_csv(args.train_path)
    dev_df = pd.read_csv(args.dev_path)
    
    tag2idx = {}
    if args.task == 'ner':
        unique_tags = set()
        for df in [train_df, dev_df]:
            for item in df['gold_ner'].dropna():
                if isinstance(item, str):
                    try: data = json.loads(item)
                    except json.JSONDecodeError: data = ast.literal_eval(item)
                else:
                    data = item
                for x in data:
                    clean = str(x[1]).replace("+NEGATION", "").replace("QUANTITY", "").replace("TEMPORALITY", "").split("-")[-1].strip()
                    if clean.lower() == 'o' or clean == "":
                        clean = "O"
                    unique_tags.add(clean)
        tag2idx = {tag: idx for idx, tag in enumerate(sorted(list(unique_tags)))}
        
        os.makedirs("/gscratch6/users/avarela/SDOH/benchmarking", exist_ok=True)
        with open(f"/gscratch6/users/avarela/SDOH/benchmarking/{args.model_key}_{args.task}_tags.json", "w") as f:
            json.dump(tag2idx, f)

    PARAM_GRID = [
        {"lr": 2e-5, "epochs": 5},
        {"lr": 3e-5, "epochs": 5},
        {"lr": 5e-5, "epochs": 5},
        {"lr": 2e-5, "epochs": 10},
        {"lr": 3e-5, "epochs": 10},
        {"lr": 5e-5, "epochs": 10},
    ]
    
    best_overall_f1 = -1.0
    best_hyperparameters = None
    
    print(f"=== Starting Hyperparameter Tuning Matrix for Model: {args.model_key} ===")
    
    for trial_idx, params in enumerate(PARAM_GRID):
        print(f"\n--- Running Trial {trial_idx + 1}/{len(PARAM_GRID)}: LR={params['lr']}, Epochs={params['epochs']} ---")
        
        if args.task == 'classification':
            train_ds = ClassificationDataset(train_df, tokenizer)
            dev_ds = ClassificationDataset(dev_df, tokenizer)
        else:
            train_ds = MultiModifierNERDataset(train_df, tokenizer, tag2idx)
            dev_ds = MultiModifierNERDataset(dev_df, tokenizer, tag2idx)
            
        train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
        dev_loader = DataLoader(dev_ds, batch_size=8)
        
        models, optimizers, schedulers = {}, {}, {}
        steps = len(train_loader) * params["epochs"]
        
        if args.task == 'classification':
            models['cls'] = MultiHeadClassifier(model_name).to(device)
            optimizers['cls'] = torch.optim.AdamW(models['cls'].parameters(), lr=params['lr'])
            schedulers['cls'] = get_linear_schedule_with_warmup(optimizers['cls'], 0, steps)
        else:
            models['ner'] = MultiHeadTokenClassifier(model_name, num_base_labels=len(tag2idx)).to(device)
            optimizers['ner'] = torch.optim.AdamW(models['ner'].parameters(), lr=params['lr'])
            schedulers['ner'] = get_linear_schedule_with_warmup(optimizers['ner'], 0, steps)
                
        trial_best_f1 = -1.0
        patience, patience_counter = 2, 0
        
        for epoch in range(params["epochs"]):
            train_loss = train_epoch(models, train_loader, optimizers, schedulers, device, args.task)
            dev_f1 = eval_model(models, dev_loader, device, args.task)
            print(f"Epoch {epoch+1} | Trial Loss: {train_loss:.4f} | Dev Macro-F1: {dev_f1:.4f}")
            
            if dev_f1 > trial_best_f1:
                trial_best_f1 = dev_f1
                patience_counter = 0
                
                if trial_best_f1 > best_overall_f1:
                    best_overall_f1 = trial_best_f1
                    best_hyperparameters = params
                    
                    save_dir = "/gscratch6/users/avarela/SDOH/benchmarking"
                    os.makedirs(save_dir, exist_ok=True)
                    
                    if args.task == 'classification':
                        save_path = f"{save_dir}/{args.model_key}_classification_best.pt"
                        if os.path.exists(save_path):
                            os.remove(save_path)
                        torch.save(models['cls'].state_dict(), save_path)
                    else:
                        save_path = f"{save_dir}/{args.model_key}_ner_shared_best.pt"
                        if os.path.exists(save_path):
                            os.remove(save_path)
                        torch.save(models['ner'].state_dict(), save_path)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break
                    
    print(f"\n=======================================================")
    print(f"TUNING COMPLETE for {args.model_key} [{args.task}]")
    print(f"Best Trial Dev Set Combined Macro-F1 Score: {best_overall_f1:.4f}")
    print(f"Winning Parameters: {best_hyperparameters}")
    print(f"=======================================================")

if __name__ == "__main__":
    main()