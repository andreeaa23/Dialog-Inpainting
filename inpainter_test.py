from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

trained_model_path = 'fine_tuned_T5'
trained_tokenizer_path = 'fine_tuned_T5_tokenizer'

model = T5ForConditionalGeneration.from_pretrained(trained_model_path).to(device)
tokenizer = T5Tokenizer.from_pretrained(trained_tokenizer_path)


# Inference/Predictions
def generate_question(model, tokenizer, dialog, device):

    input_text = "dialog: " + dialog
    # Encode the dialog
    max_len = 512
    encoded_input = tokenizer.encode_plus(input_text, max_length=max_len, padding='max_length', truncation=True,return_tensors="pt")
    input_ids = encoded_input["input_ids"].to(device)
    attention_mask = encoded_input["attention_mask"].to(device)

    model.eval()
    output = model.generate(
                   input_ids=input_ids,
                   attention_mask=attention_mask,
                   early_stopping=True,
                   num_beams=5,
                   num_return_sequences=1,
                   no_repeat_ngram_size=2,
                   max_length=72
                   )
    
    # Decode and print the output
    #question = tokenizer.decode(output[0], skip_special_tokens=True)
    decoded = [tokenizer.decode(ids, skip_special_tokens=True) for ids in output]
    masked_question = decoded[0].replace("question:", "")
    masked_question = masked_question.strip()
    question = masked_question
    
    return question

def fill_masks_autoregressively(model, tokenizer, partial_dialog, device):
    
    chunks = [chunk + '.' for chunk in partial_dialog.split('. ') if chunk]
    filled_dialog = ""
    current_chunk_index = 0
    max_len = 512

    while current_chunk_index < len(chunks):
        chunk = chunks[current_chunk_index]
        if "<extra_id_" in chunk:
            dialog = ""
            chunk_to_process = chunk.replace(f"<extra_id_{current_chunk_index}>", "<extra_id_0>")
            dialog = "dialog: " + chunk_to_process
            print("Chunk to process:", chunk_to_process)
            
            encoded_input = tokenizer.encode_plus(dialog, max_length=max_len, padding='max_length', truncation=True,return_tensors="pt")
            input_ids = encoded_input["input_ids"].to(device)
            attention_mask = encoded_input["attention_mask"].to(device)

            model.eval()
            output = model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        early_stopping=True,
                        num_beams=5,
                        num_return_sequences=1,
                        no_repeat_ngram_size=2,
                        max_length=72
                        )
    
            # Decode and print the output
            #question = tokenizer.decode(output[0], skip_special_tokens=True)
            decoded = [tokenizer.decode(ids, skip_special_tokens=True) for ids in output]
            masked_question = decoded[0].replace("question:", "")
            masked_question = masked_question.strip()
            generated_question = masked_question
            
            print("Generated question: ", generated_question)
            
            # Replace the generic mask in the original chunk with the generated question
            filled_chunk = chunk.replace(f"<extra_id_{current_chunk_index}>", generated_question)
            filled_dialog += filled_chunk
        else:
            filled_dialog += chunk
        
        current_chunk_index += 1
        if current_chunk_index < len(chunks):
            filled_dialog += " "

    return filled_dialog

dialog = "1: <extra_id_0> 0: He played for the Red Sox and the White Sox. 1: Did he play good? 0: Yes. 1: Are there any other interesting aspects about this article? 0: Yes. 1: Did he play for anymore teams 0: Yes. 1: Did he resign 0: Yes. 1: How long was he with the white sox 0: He was with the White Sox for 3 years. 1: Did he get a manager position 0: CANNOTANSWER 1: What did you find interesting in this article about Harry Hooper 0: I found it interesting that Harry Hooper holds the Red Sox franchise records for most triples and stolen bases. 1: Any other records he hold 0: He holds the Red Sox franchise records for most triples (130) and stolen bases (300) in his career. 1: What record gave him a break 0: The record that gave him a break was hitting better than.300 five times in his career. 1: What other teams did he play for if any 0: He played for the Red Sox. 1: Did he do great on the red sox team? 0: Yes."
print(dialog)
predicted_question = generate_question(model, tokenizer, dialog, device)
print(predicted_question)
print("\n")

dialog = "1: Why did he return to the WWWF? 0: He returned to the WWWF because of an agreement with promoter Vincent J. McMahon (Senior).) 1: What was his agreement with McMahon? 0: CANNOTANSWER 1: How did people respond to his return? 0: CANNOTANSWER 1: What else happened during 1977-1981? 0: In 1977, Graham defeated Bruno Sammartino for the WWWF Heavyweight Championship. 1: <extra_id_0> 0: Yes. 1: What happened after he defeated Bruno? 0: After he defeated Bruno, Graham held the title for nine and a half months. 1: Who took the title after? 0: CANNOTANSWER"
print(dialog)
predicted_question = generate_question(model, tokenizer, dialog, device)
print(predicted_question)
print("\n")

partial_dialog = "1: <extra_id_0> 0: Simona Halep (Romanian pronunciation: [siˈmona haˈlep];[3] born 27 September 1991) is a Romanian professional tennis player. 1: <extra_id_1> 0: She has been ranked world number one in singles twice between 2017 and 2019, for a total of 64 weeks, which ranks twelfth in the history of the Women's Tennis Association (WTA) rankings. 1: <extra_id_2> 0: Halep was the year-end No. 1 in 2017 and 2018. 1: <extra_id_3> 0: She has won two Grand Slam singles titles: the 2018 French Open and the 2019 Wimbledon Championships. 1: <extra_id_4> 0: From 27 January 2014 to 8 August 2021, Halep was ranked in the top 10 for 373 consecutive weeks, the eighth-longest streak in WTA history. 1: <extra_id_5> 0: During this seven-year span, she finished each year ranked no lower than No. 4."
document_title = "Simona Halep"

filled_dialog = fill_masks_autoregressively(model, tokenizer, partial_dialog,  device)
print(filled_dialog)

# dialog = "1: Who is Simona Halep? 0: Simona Halep (Romanian pronunciation: [siˈmona haˈlep];[3] born 27 September 1991) is a Romanian professional tennis player. 1: <extra_id_0> 0: She has been ranked world number one in singles twice between 2017 and 2019, for a total of 64 weeks, which ranks twelfth in the history of the Women's Tennis Association (WTA) rankings."
# predicted_question = generate_question(model, tokenizer, dialog, device)
# print(predicted_question)