from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch


def get_question(sentence, answer, mdl, tknizer):
    
    text = "context: {} answer: {}".format(sentence, answer)
    print(text)
    max_len = 256
    encoding = tknizer.encode_plus(text, max_length=max_len, padding='max_length', truncation=True, return_tensors="pt")

    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    outs = mdl.generate(input_ids=input_ids,
                        attention_mask=attention_mask,
                        early_stopping=True,
                        num_beams=5,
                        num_return_sequences=1,
                        no_repeat_ngram_size=2,
                        max_length=72)

    dec = [tknizer.decode(ids, skip_special_tokens=True) for ids in outs]

    question = dec[0].replace("question:", "")
    question = question.strip()
    return question

trained_model_path = 't5_trained_model2'
trained_tokenizer_path = 't5_tokenizer2'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = T5ForConditionalGeneration.from_pretrained(trained_model_path).to(device)
tokenizer = T5Tokenizer.from_pretrained(trained_tokenizer_path)

# Examples
context = "Donald Trump is an American media personality and businessman who served as the 45th president of the United States."
answer = "Donald Trump"
ques = get_question(context, answer, model, tokenizer)
print("Question:", ques)

context = " Since its topping out in 2013, One World Trade Center in New York City has been the tallest skyscraper in the United States."
answer = "World Trade Center"
ques = get_question(context, answer, model, tokenizer)
print("Question:", ques)
