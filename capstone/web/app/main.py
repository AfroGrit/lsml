from typing import Optional

from fastapi import FastAPI
import config
import torch
from model import model


def ready_model():
    # TODO - documentation
    # Load a trained model and vocabulary that you have fine-tuned
    talker = model.from_pretrained(config.MODEL_PATH)
    tokenizer = config.TOKENIZER.from_pretrained(config.MODEL_PATH)

    # Copy the model to the GPU.
    talker.to(config.DEVICE)
    return talker


def conversation(question):
    talker = ready_model()
    # Put model in evaluation mode
    talker.eval()

    # Tracking variables
    predictions, true_labels = [], []

    # Predict
    # Add question to GPU
    question = tuple(t.to(config.DEVICE) for t in question)

    # Unpack the inputs from our dataloader
    b_input_ids, b_input_mask, b_labels = question

    # Telling the model not to compute or store gradients, saving memory and
    # speeding up prediction
    with torch.no_grad():
        # Forward pass, calculate logit predictions.
        result = talker(b_input_ids,
                        token_type_ids=None,
                        attention_mask=b_input_mask,
                        return_dict=True)

    logits = result.logits

    # Move logits and labels to CPU
    logits = logits.detach().cpu().numpy()
    label_ids = b_labels.to('cpu').numpy()

    # Store predictions and true labels
    predictions.append(logits)
    true_labels.append(label_ids)

    print('    DONE.')


app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World's apart"}


# @app.get("/items/{item_id}")
# def read_item(item_id: int, q: Optional[str] = None):
#     return {"item_id": item_id, "q": q}


@app.get("/talk/{sentence}")
def predict(sentence: str):
    return {"message": str(conversation(sentence)[0][0])}
