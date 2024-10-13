import random
import json
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, logging as transformers_logging
from context_manager import ContextManager
import logging
from safetensors.torch import load_file

# Suppress transformers warnings
transformers_logging.set_verbosity_error()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load intents
with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

# Load pre-trained model and tokenizer
model_name = 'bert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)

# Adjust the model's classifier to match the number of intents
model.classifier = torch.nn.Linear(model.config.hidden_size, len(intents['intents'])).to(device)

# Load weights from model.safetensors
weights = load_file('C:\\Users\\rehan\\Downloads\\Compressed\\pytorch-chatbot-master\\pytorch-chatbot-master\\fine_tuned_model\\model.safetensors')
model.load_state_dict(weights)

bot_name = "Sam"
context_manager = ContextManager()

logging.basicConfig(level=logging.DEBUG, filename='chatbot.log', filemode='a',
                    format='%(asctime)s - %(levelname)s - %(message)s')

def log_interaction(user_id, user_input, bot_response):
    logging.info(f"User ID: {user_id}, User Input: {user_input}, Bot Response: {bot_response}")

def get_feedback():
    feedback = input("Was this response helpful? (yes/no): ")
    return feedback.lower() == 'yes'

def get_response(sentence, user_id):
    inputs = tokenizer(sentence, return_tensors='pt').to(device)
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_id = logits.argmax().item()
    tag = intents['intents'][predicted_class_id]['tag']

    probs = torch.softmax(logits, dim=1)
    prob = probs[0][predicted_class_id]

    # Debugging statements
    logging.debug(f"Input Sentence: {sentence}")
    logging.debug(f"Predicted class ID: {predicted_class_id}")
    logging.debug(f"Tag: {tag}")
    logging.debug(f"Probability: {prob.item()}")

    if prob.item() > 0.1:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                logging.debug(f"Matched Intent: {intent['tag']}")
                if 'context_set' in intent:
                    context_manager.add_to_context(user_id, 'context', intent['context_set'])
                if 'context_filter' not in intent or context_manager.get_from_context(user_id, 'context') == intent['context_filter']:
                    response = random.choice(intent['responses'])
                    logging.debug(f"Response: {response}")
                    return response
    logging.debug("No matching intent found or probability too low.")
    return "I'm not sure how to respond to that. Can you please rephrase or ask something else?"

def chat():
    print("Let's chat! (type 'quit' to exit)")
    user_id = "default_user"

    while True:
        sentence = input("You: ")
        if sentence.lower() == "quit":
            break

        resp = get_response(sentence, user_id)
        print(f"{bot_name}: {resp}")
        log_interaction(user_id, sentence, resp)

        if not get_feedback():
            print(f"{bot_name}: I'm sorry. How can I improve?")

if __name__ == "__main__":
    chat()