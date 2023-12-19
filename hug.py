from datasets import load_dataset
from transformers import AutoTokenizer

raw_datasets = load_dataset("squad")

# # DatasetDict({
# #     train: Dataset({
# #         features: ['id', 'title', 'context', 'question', 'answers'],
# #         num_rows: 87599
# #     })
# #     validation: Dataset({
# #         features: ['id', 'title', 'context', 'question', 'answers'],
# #         num_rows: 10570
# #     })
# # })

# # raw_datasets

# print("Context: ", raw_datasets["train"][0]["context"])
# print("Query: ", raw_datasets["train"][0]["question"])
# # answer e format din: ansewr text si answer start(de unde incepe rasp din context)
# print("Answer: ", raw_datasets["train"][0]["answers"])

# # !! in timpul antrenarii poate exista un singur raspuns valid,
# # outem retesta cu Dataset.filter() method
# raw_datasets["train"].filter(lambda x: len(x["answers"]["text"]) != 1)

# # Dataset({
# #     features: ['id', 'title', 'context', 'question', 'answers'],
# #     num_rows: 0
# # })

# # !! pentru evaluare pot exista mai multe rasp posibile pentru fiecare intrebare
# # care pot fi identice sau diferite

# print("----------\n")
# print(raw_datasets["validation"][0]["answers"])
# print(raw_datasets["validation"][2]["answers"])

# #!! unele intrebari pot avea mai multe raso posibile si script ul
# # va compara un rasp predictibil cu toate rasp acceptabile si va lua
# # cel mai bun scor

# print("----------\n")
# print(raw_datasets["validation"][2]["context"])
# print(raw_datasets["validation"][2]["question"])


# # !! convertire text din input in ID-uri
# # trb sa ii facem un fine-tuning cu modelul BERT(sau oricare altul
# # care are tokenizare rapida)

# print("****************\n")
model_checkpoint = "bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
# # tokenizer.is_fast

# # trimitere intrebare la tokenizer si context si el va introduce
# # tokeni speciale pentru a crea o prop de genul: [CLS] question [SEP] context [SEP]
# context = raw_datasets["train"][0]["context"]
# question = raw_datasets["train"][0]["question"]

# # inputs = tokenizer(question, context)
# # tokenizer.decode(inputs["input_ids"])

# # label-urile vor fi apoi indecsii token-urilor de start si end
# # a raspunsurilor si modelul va fi facut sa prezica un start si un end
# # logic per token la input
# # in ex de mai sus, context-ul nu e ft lung, dar unele exemple din dataset
# # au context foarte lung care depasesc lungimea maxima setate(384 e setata de ei)
# # In cap 6 putem explora interiorul pipeline-urilor de question answers,
# # ne vom confrunta cu context-uri lungi prin crearea mai multor
# # features de training uri pentru un sample din dataset ul nostru
# # cu o fereastra  pe slide intre ele?
# # Pt a vedea cum functioneaza putem seta exemplul curent limitand
# # lungimea la 100 si adaugand o fereastra cu 50 tokens


# inputs = tokenizer(
#     question,
#     context,
#     max_length=100,
#     # truncate the context (which is in the second position) when the question with its context is too long
#     truncation="only_second",
#     # set the number of overlapping tokens between two successive chunks (here 50)
#     stride=50,
#     # let the tokenizer know we want the overflowing tokens
#     return_overflowing_tokens=True,
# )

# # for ids in inputs["input_ids"]:
# #     print(tokenizer.decode(ids))

# #   Rezulta textul impartit in 4 intrari, fiecare continand o intrebare si o parte din context. De obs ca
# # rasp la intrebarea  (“Bernadette Soubirous”) apare doar in a treia si ultima intrare, deci
# # confruntandu ne cu contexturi lungi in acest mod vom crea niste exemple de training unde rasp
# # nu e inclus in context. Pt aceste exemple, etichetele vor fi start_position=end_position=0(deci
# # noi prezicem token ul[CLS]). Vom seta aceste etichete in cazul nefericit cand rasp ar fi trunchiat
# # si avem doar start(sau end). Pt exemplele in care rasp e full in context, etichetele
# # vor fi indexul token-ului unde incepe rasp si indexul token-ului unde se termina rasp.
# #   Datasetul ne da primul caracter de start din context, si adaugand lungimea raspunsului putem
# # gasi caracterul final din context. Pentru a mapa asta in indicii token-urilor, noi trebuie
# # sa folosim maparile offset studiate in Cap6. Putemk face ca tokenizatorul sa ne returneze acestea
# # prin pasarea argumentului  return_offsets_mapping=True:

# inputs = tokenizer(
#     question,
#     context,
#     max_length=100,
#     truncation="only_second",
#     stride=50,
#     return_overflowing_tokens=True,
#     return_offsets_mapping=True,
# )
# # dict_keys(['input_ids', 'token_type_ids', 'attention_mask', 'offset_mapping', 'overflow_to_sample_mapping'])
# # inputs.keys()
# # Se obtine o lista ce contine: IDs, IDS of token type, attention mask, offset mask pe care am cerut-o
# # si o cheie extra numita overflow_to_sample_mapping. Val corespunzatoare va fi folosita cand
# # tokenizam mult text in acelasi timp. De cand o prop poate da mai multe caracteristici, ea va mapa
# # fiecare caractersitica la exemplul din care provine.
# # !!! Ptc noi tokenizam un singur exemplu, vom obtine o lista de zero-uri.

# # print(inputs["overflow_to_sample_mapping"])  # [0, 0, 0, 0]

# #   Dar daca tokenizam mai multe propozitii va fi mult mai folositor
# print("****************\n")
inputs = tokenizer(
    raw_datasets["train"][2:6]["question"],
    raw_datasets["train"][2:6]["context"],
    max_length=100,
    truncation="only_second",
    stride=50,
    return_overflowing_tokens=True,
    return_offsets_mapping=True,
)

# print(f"The 4 examples gave {len(inputs['input_ids'])} features.")
# print(
#     f"Here is where each comes from: {inputs['overflow_to_sample_mapping']}.")

# # Putem obs ca primele 3 exemple(de  la indicii 2,3,4 din setuk de train) fiecare da 4 caracteristici
# # si ultimul exemplu(de la indexul 5 din setul de training) da 7 caracterestici.
# # Info aceasta va fi folositoare pentru maparea fiecarei caract pe care o obtinem cu
# # eticheta sa corepsuncatoare.
# #           Etichetele sunt:
# # 1. (0,0) = rasp nu e in context
# # 2. (start_position, end_position) = rasp e in context, unde start_index este indexul
# # tokenului (din input IDs) de start al rasp si end_position este indexul de sf.
# # Pentru a determina in care caz ne incadram si pozitiile din tokens, noi prima data trebuie sa gasim
# # indici de start si end din context pentru input IDs. Putem folosi token type IDs pentru a face asta, dar de cand
# # asta nu exista oentru toate modelele(cum ar fi DistilBERT) vom folosi in loc metoda sequence_ids()  a lui BatchEncoding
# # ca sa ne returneze tokenizatorul nostru.
# #   Dupa ce obtinem acesti indici pentru tokens, ne uitam la offseturile coresp  care sunt tupluri de 2 integeri
# # ce reprezinta span-ul caracterelor din interiorul contextului original. Putem detecta astfel daca chunk-ul din context
# # in aceasta caracteristica incepe dupa ce raspuns sau se termina inainte ca rasp sa inceapa(eticheta in cazul asta e (0,0)).
# # Daca nu e in cazul asta, vom cauta intr-o bucla pana gasim primul sau ultinmul token din raspuns.
# answers = raw_datasets["train"][2:6]["answers"]
# start_positions = []
# end_positions = []

# for i, offset in enumerate(inputs["offset_mapping"]):
#     sample_idx = inputs["overflow_to_sample_mapping"][i]
#     answer = answers[sample_idx]
#     start_char = answer["answer_start"][0]
#     end_char = answer["answer_start"][0] + len(answer["text"][0])
#     sequence_ids = inputs.sequence_ids(i)

#     # Find the start and end of the context
#     idx = 0
#     while sequence_ids[idx] != 1:
#         idx += 1
#     context_start = idx
#     while sequence_ids[idx] == 1:
#         idx += 1
#     context_end = idx - 1

#     # If the answer is not fully inside the context, label is (0, 0)
#     if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
#         start_positions.append(0)
#         end_positions.append(0)
#     else:
#         # Otherwise it's the start and end token positions
#         idx = context_start
#         while idx <= context_end and offset[idx][0] <= start_char:
#             idx += 1
#         start_positions.append(idx - 1)

#         idx = context_end
#         while idx >= context_start and offset[idx][1] >= end_char:
#             idx -= 1
#         end_positions.append(idx + 1)

# print(start_positions, end_positions)

# # Aruncam o privire acum asupra rezultatelor pt a verifica daca ce am facut este corect.
# # Pentru aceasta caracteristica am gasit (83, 85) ca etichete, deci sa comparam acum
# # rasp teoretice cu span-ul decodat al tokenurilor de la 83 la 85(inclus):
# idx = 0
# sample_idx = inputs["overflow_to_sample_mapping"][idx]
# answer = answers[sample_idx]["text"][0]

# start = start_positions[idx]
# end = end_positions[idx]
# labeled_answer = tokenizer.decode(inputs["input_ids"][idx][start: end + 1])

# print(f"Theoretical answer: {answer}, labels give: {labeled_answer}")

# # incercam si pt index=4 unde am setat etichetele (0,0) ->rasp nu e in chunk ul de context al acelei caracteristici
# idx = 4
# sample_idx = inputs["overflow_to_sample_mapping"][idx]
# answer = answers[sample_idx]["text"][0]

# decoded_example = tokenizer.decode(inputs["input_ids"][idx])
# print(f"Theoretical answer: {answer}, decoded example: {decoded_example}")
# intr-adevar, nu vedem raps inauntrul contextului
# acum ca am vazut pas cu pas cum sa preprocesam data trainingul nostru, putem grupa totul inr-o functie
# care sa se aplice asupra intregului dataset. Vom umple fiecare caracteristica cu maximul lungimii pe care
# l am setat, pt ca mult context e lung(si prop coresp vor fi impartite in multe caracteristici), deci
# nu e niciun beneficiu real pt a aplica padding dinamic aici
max_length = 384
stride = 128


def preprocess_training_examples(examples):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=max_length,
        truncation="only_second",
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    offset_mapping = inputs.pop("offset_mapping")
    sample_map = inputs.pop("overflow_to_sample_mapping")
    answers = examples["answers"]
    start_positions = []
    end_positions = []

    for i, offset in enumerate(offset_mapping):
        sample_idx = sample_map[i]
        answer = answers[sample_idx]
        start_char = answer["answer_start"][0]
        end_char = answer["answer_start"][0] + len(answer["text"][0])
        sequence_ids = inputs.sequence_ids(i)

        # Find the start and end of the context
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        # If the answer is not fully inside the context, label is (0, 0)
        if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            # Otherwise it's the start and end token positions
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs


# Am definit 2 constante caee sa determina lung maxima folsita ca si lungimea pt fereastra
# glisanta si am adaugat un bit tiny pentru curatare inainte de tokenizare pentru ca unele
# intrebari din acest dataset au spatii extra la inceput si final care nu adauga nimic asa ca
# le am eliminat ca erau in plus si degeaba
# Pentru a adauga aceasta functie la intreg datasetul de trainign, folsim metoda Dataset.map()
# cu flag-ul batched=True. E necesar aici pentru ca modificam lungimea dataset-ului(un singur exemplu
# poate da mai multe caracteristici de training)
train_dataset = raw_datasets["train"].map(
    preprocess_training_examples,
    batched=True,
    remove_columns=raw_datasets["train"].column_names,
)
len(raw_datasets["train"]), len(train_dataset)
# (87599, 88729) obs ca se adauga 1000 de caract in plus.

# PREPROCESARE DATE DE VALIDARE
# Nu trb sa generam etichete. Trb sa interpretam predictiile modelului in intervale(spans)
# ale contextului original. Pt asta, trb sa stocam atat offset mappings cat si modul in care
# facem legatura intre fiecare caracteristica creata pentru exemplul original din care vine.
# Pentru ca exista o coloana ID in dataset ul original, vom folosi acest ID.
# Singurul lucru pe care il adaugam aici este un  bit tiny pentru curatarea offset mappings.
# Contin offset-uri pentru intrebare si context, dar odata ce noi suntem in stagiul de post-procesare
# nu avem o modalitate prin care sa stim care parte din input IDs corespunde cu contextul si care parte
# din intrebare (metoda sequence_ids pe care noi am folosit-o este disponibila numai pt outputul tokenizerului).
# Deci vom seta offsetul corespunzator intrebarii ca fiinnd None


def preprocess_validation_examples(examples):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=max_length,
        truncation="only_second",
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_map = inputs.pop("overflow_to_sample_mapping")
    example_ids = []

    for i in range(len(inputs["input_ids"])):
        sample_idx = sample_map[i]
        example_ids.append(examples["id"][sample_idx])

        sequence_ids = inputs.sequence_ids(i)
        offset = inputs["offset_mapping"][i]
        inputs["offset_mapping"][i] = [
            o if sequence_ids[k] == 1 else None for k, o in enumerate(offset)
        ]

    inputs["example_id"] = example_ids
    return inputs


validation_dataset = raw_datasets["validation"].map(
    preprocess_validation_examples,
    batched=True,
    remove_columns=raw_datasets["validation"].column_names,
)
len(raw_datasets["validation"]), len(validation_dataset)

# obs ca se adauga doar cateva sute deci contextul in datasetul de validare este mai micut
# Pana aici am preprocesat toate datele, urmeaza sa antrenam acum.


# Fine-tuning the model with Keras
