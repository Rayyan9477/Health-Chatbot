import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AdamW, get_linear_schedule_with_warmup

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ChatDataset(Dataset):
    def __init__(self, data, tokenizer, max_len):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        inputs = self.tokenizer.encode_plus(
            item['pattern'],
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=False,
            truncation=True
        )
        return {
            'input_ids': torch.tensor(inputs['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(inputs['attention_mask'], dtype=torch.long),
            'labels': torch.tensor(item['label'], dtype=torch.long)
        }

def train_model():
    with open('intents.json', 'r') as f:
        intents = json.load(f)

    data = []
    for idx, intent in enumerate(intents['intents']):
        for pattern in intent['patterns']:
            data.append({'pattern': pattern, 'label': idx})

    model_name = 'bert-base-uncased'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(intents['intents'])).to(device)

    dataset = ChatDataset(data, tokenizer, max_len=128)
    train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

    optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
    total_steps = len(train_loader) * 3  # 3 epochs
    loss_fn = torch.nn.CrossEntropyLoss().to(device)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    print(f"Training on {device}")

    for epoch in range(3):
        print(f"Epoch {epoch + 1}/{3}")

    model.train()
    for epoch in range(3):
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()

    model.save_pretrained('fine_tuned_model')
    tokenizer.save_pretrained('fine_tuned_model')

if __name__ == "__main__":
    train_model()