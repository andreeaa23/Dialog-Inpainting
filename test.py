from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

trained_model_path = 't5_trained_model2'
trained_tokenizer_path = 't5_tokenizer2'

model = T5ForConditionalGeneration.from_pretrained(trained_model_path).to(device)
tokenizer = T5Tokenizer.from_pretrained(trained_tokenizer_path)

# Inference/Predictions
def get_question(context,  model, tokenizer, answer=""):
    
    text = "context: {} answer: {}".format(context, answer)
    #print(text)
    
    max_len = 512
    encoding = tokenizer.encode_plus(text, max_length=max_len, padding='max_length', truncation=True, return_tensors="pt")

    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    model.eval()
    beam_outputs = model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        early_stopping=True,
                        num_beams=5,
                        num_return_sequences=1,
                        no_repeat_ngram_size=2,
                        max_length=72 #max len for the generated question
                        )

    #decode the generated mask question
    decoded = [tokenizer.decode(ids, skip_special_tokens=True) for ids in beam_outputs]
    # question = decoded[0].replace("question:", "")
    # question = question.strip()
    masked_question = decoded[0].replace("question:", "")
    masked_question = masked_question.strip()
    question = masked_question
    
    return question

# Examples
context = "Donald Trump is an American media personality and businessman who served as the 45th president of the United States."
answer = "Donald Trump"
question = get_question(context, model, tokenizer, answer)
print("Question:", question)

context = "Since its topping out in 2013, One World Trade Center in New York City has been the tallest skyscraper in the United States."
answer = "World Trade Center"
question = get_question(context, model, tokenizer, answer)
print("Question:", question)

context = "President Donald Trump said and predicted that some states would reopen this month."
answer = "Donald Trump"
question = get_question(context, model, tokenizer, answer)
print("Question:", question)

context = "Simona Halep (Romanian pronunciation: [siˈmona haˈlep];[3] born 27 September 1991) is a Romanian professional tennis player."
answer = "professional tennis player"
question = get_question(context,  model, tokenizer, answer)
print("Question:", question)