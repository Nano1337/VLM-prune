import os
import pandas as pd
from PIL import Image
from tqdm import tqdm


import torch 
from torch.utils.data import DataLoader, Dataset
from transformers import AutoProcessor, AutoModelForCausalLM

import torch.nn.functional as F

parquet_file_path = "/home/haoli/VLM-prune/tenk_filtered.parquet"
# parquet_file_path = "/home/haoli/VLM-prune/tenk_filtered_english.parquet"
# parquet_file_path = "/home/haoli/VLM-prune/tenk_filtered_english_dense_captions.parquet"
# parquet_file_path = "/home/haoli/VLM-prune/tenk_filtered_generated_sentences_2words.parquet"
image_dir = "data/sampled_datacomp/sampled_datacomp_images"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model_id = "microsoft/Florence-2-base-ft" # NOTE: must use fine-tuned model (base or large)
model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-base-ft", trust_remote_code=True, revision='refs/pr/6')
model.to(device)
processor = AutoProcessor.from_pretrained("microsoft/Florence-2-base-ft", trust_remote_code=True, revision='refs/pr/6')

# model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-large-ft", trust_remote_code=True, revision='refs/pr/10').to(device)
# processor = AutoProcessor.from_pretrained("microsoft/Florence-2-large-ft", trust_remote_code=True, revision='refs/pr/10')

# model = AutoModelForCausalLM.from_pretrained("HuggingFaceM4/Florence-2-DocVQA", trust_remote_code=True).to(device)
# processor = AutoProcessor.from_pretrained("HuggingFaceM4/Florence-2-DocVQA", trust_remote_code=True)
model.eval()
# default_question_template = 'Does this figure show "{}"? Please answer yes or no.'
# default_question_template = 'Is the caption "{}" accurate and detailed in describing the image?\n Please answer using a single word: yes or no.'
# default_answer_template = "yes" # for disagreement, use "No"

default_question_template = 'What does the image describe? The image shows:{}'


class TenkDataset(Dataset):
    def __init__(self):
        self.df = pd.read_parquet(parquet_file_path)
        self.image_dir = image_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        uid = self.df.iloc[idx]['uid']
        key = self.df.iloc[idx]['key']
        image_path = os.path.join(self.image_dir, key + ".jpg")
        caption = self.df.iloc[idx]['caption'] # CHANGE: generated_random_sentence
        return image_path, caption, uid
    
def main(): 

    # NOTE: there's a max seq length of 1024
    batch_size = 128 # FIXME: see test case at bottom for why we can't do batched inference yet

    dataset = TenkDataset()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=32)
    results = []

    for batch in tqdm(loader, desc="Processing batches", unit="batch"):
        image_paths, captions, uids = batch
        captions = list(captions)
        # try: 
        
        import time
        start_time = time.time()
        scores = forward_debug(image_paths, captions) # forward pass here
        end_time = time.time()
        print(f"Forward pass took {end_time - start_time} seconds")
        exit()
        # except Exception as e:
        #     print(f"Error occurred: {e}")
        #     print("Image paths that caused the error:")
        #     for path in image_paths:
        #         print(path)
        #         try:
        #             file_size = os.path.getsize(path)
        #             print(f"File size: {file_size} bytes")
        #         except OSError as e:
        #             print(f"Could not get file size for {path}: {e}")
        #     scores = torch.ones(len(image_paths)) * -1 # filter out these invalid samples later
        if not isinstance(scores, list):
            scores = scores.tolist()
        batch_results = list(zip(uids, scores))
        results.extend(batch_results)
        del scores, batch_results

    # Create a DataFrame from the results
    df = pd.DataFrame(results, columns=['uid', 'VQAscore'])

    # Save the DataFrame to a Parquet file
    df.to_parquet('fps_base_10k_normal_generated.parquet', index=False) # CHANGE: output results path

# def inference_forward(image_paths, texts):

#     texts = [default_question_template.format(text) for text in texts]
#     images = [Image.open(image_path).convert("RGB") for image_path in image_paths]

#     inputs = processor(text=texts, images=images, return_tensors="pt", padding=True, truncation=True, max_length=1024)

#     # find a way to do this more efficiently without having to re-encode the image
#     # "yes" is input_id 10932 and "Yes" is input_id 9904
#     # answer_input = processor(text=default_answer_template, images=image, return_tensors="pt")

#     generated_ids = model.generate(
#         input_ids=inputs["input_ids"].to(device), 
#         pixel_values=inputs["pixel_values"].to(device), 
#         max_new_tokens=4, 
#         num_beams=1, # use greedy decoding
#         early_stopping=False, 
#         do_sample=False, 
#         output_scores=True, # output_logits doesn't work for this model
#         return_dict_in_generate=True,
#     )
    
#     # print(generated_ids.keys())
#     # print(generated_ids['sequences'])
#     # print(generated_ids['scores'][0].shape)

#     scores = generated_ids['scores']
#     scores = torch.stack(scores).permute(1, 0, 2)
#     scores = scores[:, -2]
#     scores = F.softmax(scores, dim=-1)
#     scores = scores[:, 10932]

#     return scores

#     ### code for actual inference, NOTE: set return_dict_in_generate=False to use this
#     # generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
#     # parsed_answer = processor.post_process_generation(
#     #     generated_text, 
#     #     task="<CAPTION>", 
#     #     image_size=(image.width, image.height)
#     # )
#     # parsed_answer = parsed_answer['<CAPTION>']
#     # return parsed_answer 


# def forward(image_paths, texts):

#     ### Previous semi-working solution, had padding issues

#     texts = [default_question_template.format(text) for text in texts]
#     images = [Image.open(image_path).convert("RGB") for image_path in image_paths]

#     inputs = processor(text=texts, images=images, return_tensors="pt", padding=True, truncation=True, max_length=1024)

#     # find a way to do this more efficiently without having to re-encode the image
#     # "yes" is input_id 10932 and "Yes" is input_id 9904
#     # answer_input = processor(text=default_answer_template, images=image, return_tensors="pt")

#     generated_ids = model.generate(
#         input_ids=inputs["input_ids"].to(device), 
#         pixel_values=inputs["pixel_values"].to(device), 
#         max_new_tokens=100, 
#         num_beams=1, # use greedy decoding
#         early_stopping=True, 
#         do_sample=False, 
#         output_scores=True, # output_logits doesn't work for this model
#         return_dict_in_generate=True,
#         bad_words_ids=[[0]]
#     )
    
#     # print(generated_ids.keys())
#     # print(generated_ids['sequences'])
#     # print(generated_ids['scores'][0].shape)

#     scores = generated_ids['scores']
#     scores = torch.stack(scores).permute(1, 0, 2)
#     scores = scores[:, -2]
#     scores = F.softmax(scores, dim=-1)
#     scores = scores[:, 10932]

#     return scores

# def forward(image_paths, texts):  

    """
    Current working implementation of VQAscore using the florence-2 model, P(yes) doesn't work as well tho
    """
    
#     # do this in the Dataset instead to parallelize workers
#     texts = [default_question_template.format(text) for text in texts]
#     images = [Image.open(image_path).convert("RGB") for image_path in image_paths] 

#     inputs = processor(text=texts, images=images, return_tensors="pt", padding="max_length", truncation=True, max_length=1024)
#     input_ids = inputs.input_ids.to(device)
#     pixel_values = inputs.pixel_values.to(device)
#     attention_mask = inputs.attention_mask.to(device)

#     answers = [default_answer_template] * len(texts)
#     neg_answers = ["no"] * len(texts)
#     labels = processor.tokenizer(text=answers, return_tensors="pt", padding=True, return_token_type_ids=False).input_ids.to(device)
#     neg_labels = processor.tokenizer(text=neg_answers, return_tensors="pt", padding=True, return_token_type_ids=False).input_ids.to(device)
#     # find a way to do this more efficiently without having to re-encode the image
#     # "yes" is input_id 10932 and "Yes" is input_id 9904
#     # answer_input = processor(text=default_answer_template, images=image, return_tensors="pt")

#     # extract the answer token
#     outputs = model(input_ids=input_ids, pixel_values=pixel_values, attention_mask=attention_mask, labels=labels)# .logits
#     logits = outputs.logits[:, -2, :]
#     labels = labels[:, -2]
#     neg_labels = neg_labels[:, -2]
#     loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
#     loss_fct_neg = torch.nn.CrossEntropyLoss(reduction='none')
#     lm_loss = loss_fct(logits, labels)

#     neg_loss = loss_fct_neg(logits, neg_labels)
#     lm_prob = (-lm_loss).exp()
#     # neg_prob = (-neg_loss).exp()
#     # lm_prob = (lm_prob + neg_prob) / 2

#     # NOTE: original unvectorized form, requires CE loss with reduction='mean'
#     # lm_prob = torch.zeros(logits.shape[0])
#     # for k in range(lm_prob.shape[0]):
#     #     lm_prob[k] = (-loss_fct(logits[k], labels[k])).exp() # exp to cancel the log and get raw prob between 0 and 1

#     return lm_prob

loss_fct = torch.nn.CrossEntropyLoss(reduction="none")

# def forward(image_paths, texts):  
#     """
#     Perplexity-based implementation of VQAscore using the florence-2 model
#     """

#     # dummy labels to make the model work
#     fake_answers = ["no"] * len(texts)
#     fake_labels = processor.tokenizer(text=fake_answers, return_tensors="pt", padding=True, return_token_type_ids=False).input_ids.to(device)

#     # actual start of script
#     labels = texts.copy()
#     texts = [default_question_template.format(text) for text in texts]
#     images = [Image.open(image_path).convert("RGB") for image_path in image_paths]

#     inputs = processor(text=texts, images=images, return_tensors="pt", padding=True, truncation=True, max_length=1024)
#     input_ids = inputs.input_ids.to(device)
#     pixel_values = inputs.pixel_values.to(device)
#     attention_mask = inputs.attention_mask.to(device)

#     with torch.no_grad(): 
#         outputs = model(input_ids=input_ids, pixel_values=pixel_values, attention_mask=attention_mask) #, labels=fake_labels)

#     # Encode the original text (without the template)
#     labels = processor.tokenizer.encode(labels, add_special_tokens=False, return_tensors="pt")

#     print(f"labels shape: {labels.shape}")

#     # shift the labels
#     shifted_labels = labels[..., 1:].contiguous()

#     print(f"shifted_labels shape: {shifted_labels.shape}")

#     # shift the logits
#     print(f"outputs.logits shape: {outputs.logits.shape}")

#     exit()
#     shifted_logits = outputs.logits[..., :-1, :].contiguous()
#     print(f"shifted_logits shape: {shifted_logits.shape}")

#     shifted_logits = shifted_logits.view(-1, shifted_logits.size(-1))
    
#     print(f"shifted_labels shape: {shifted_labels.shape}")
#     print(f"shifted_logits shape: {shifted_logits.shape}")
#     exit()

#     loss = loss_fct(shifted_logits, shifted_labels)

#     # reshape the loss to the original shape
#     loss = loss.view(labels.size(0), labels.size(1) - 1)

#     # now remove the 0 values and create loss as a list of lists
#     loss_list = loss.tolist()

#     for i, entry in enumerate(loss_list):
#         # remove the 0 values
#         entry = [x for x in entry if x != 0] # across seq len if loss = 0
#         loss_list[i] = entry

#     # if any list is empty, remove it
#     loss_list = [entry for entry in loss_list if len(entry) > 0]

#     return loss_list

import time

def forward_debug(image_paths, texts):
    """
    Florence-2 Perplexity Score Implementation by Haoli Yin
    """

    start_time = time.time()

    # Prepare inputs, add space to the beginning of each text for tokenization consistency
    texts = [" " + text for text in texts] 
    texts = [default_question_template.format(text) for text in texts]
    images = [Image.open(image_path).convert("RGB") for image_path in image_paths]

    prepare_inputs_time = time.time()
    print(f"Time to prepare inputs: {prepare_inputs_time - start_time} seconds")

    # must do max length padding since attention masking in the model doesn't currently work
    inputs = processor(text=texts, images=images, return_tensors="pt", padding="max_length", truncation=True, max_length=1024)

    move_to_device_time = time.time()
    print(f"Time to process inputs: {move_to_device_time - prepare_inputs_time} seconds")

    # move to device
    input_ids = inputs.input_ids.to(device)
    pixel_values = inputs.pixel_values.to(device)
    attention_mask = inputs.attention_mask.to(device)

    find_indices_time = time.time()
    print(f"Time to move inputs to device: {find_indices_time - move_to_device_time} seconds")

    # find the caption-specific token indices for perplexity calculation
    start_idx = 11 # this value changes if the question template changes
    end_idx = (input_ids == 2).long().argmax(dim=1) # argmax_cuda doesn't support bool

    create_decoder_input_ids_time = time.time()
    print(f"Time to find caption-specific token indices: {create_decoder_input_ids_time - find_indices_time} seconds")

    # Create decoder_input_ids by shifting the input_ids right
    decoder_input_ids = input_ids.clone()
    decoder_input_ids[:, :-1] = input_ids[:, 1:]
    decoder_input_ids[:, -1] = processor.tokenizer.eos_token_id

    forward_pass_start_time = time.time()
    print(f"Time to create decoder input ids: {forward_pass_start_time - create_decoder_input_ids_time} seconds")

    # Forward pass
    with torch.no_grad():
        logits = model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
        ).logits

    forward_pass_end_time = time.time()
    print(f"Time for forward pass: {forward_pass_end_time - forward_pass_start_time} seconds")

    batch_size, seq_len, vocab_size = logits.shape
    max_length = (end_idx - start_idx).max().item()

    init_tensors_time = time.time()
    print(f"Time to get logits shape and max length: {init_tensors_time - forward_pass_end_time} seconds")

    # init tensors with -100 padding to max length for efficient batched CE loss calculation
    padded_logits = torch.full((batch_size, max_length, vocab_size), -100.0, device=device)
    padded_input_ids = torch.full((batch_size, max_length), -100, device=device)

    fill_tensors_start_time = time.time()
    print(f"Time to initialize tensors: {fill_tensors_start_time - init_tensors_time} seconds")

    # Fill in the relevant portions
    for i in range(batch_size):
        length = end_idx[i] - start_idx
        padded_logits[i, :length] = logits[i, start_idx-2:end_idx[i]-2] # account for decoder offset
        padded_input_ids[i, :length] = input_ids[i, start_idx:end_idx[i]]

    fill_tensors_end_time = time.time()
    print(f"Time to fill tensors: {fill_tensors_end_time - fill_tensors_start_time} seconds")

    # calculate per-token loss and perplexity
    loss = F.cross_entropy(padded_logits.view(-1, vocab_size), 
                           padded_input_ids.view(-1), 
                           ignore_index=-100, 
                           reduction='none')
    loss = loss.view(batch_size, -1)
    perplexity = torch.exp(loss.sum(dim=1) / (end_idx - start_idx).float())

    calculate_loss_time = time.time()
    print(f"Time to calculate loss and perplexity: {calculate_loss_time - fill_tensors_end_time} seconds")

    total_time = time.time()
    print(f"Total time for forward_debug: {total_time - start_time} seconds")

    return perplexity.tolist()


def forward(image_paths, texts):
    """
    Florence-2 Perplexity Score Implementation by Haoli Yin
    """

    # Prepare inputs, add space to the beginning of each text for tokenization consistency
    texts = [" " + text for text in texts] 
    texts = [default_question_template.format(text) for text in texts]
    images = [Image.open(image_path).convert("RGB") for image_path in image_paths]

    # must do max length padding since attention masking in the model doesn't currently work
    inputs = processor(text=texts, images=images, return_tensors="pt", padding="max_length", truncation=True, max_length=1024)

    # move to device
    input_ids = inputs.input_ids.to(device)
    pixel_values = inputs.pixel_values.to(device)
    attention_mask = inputs.attention_mask.to(device)

    # find the caption-specific token indices for perplexity calculation
    start_idx = 11 # this value changes if the question template changes
    end_idx = (input_ids == 2).long().argmax(dim=1) # argmax_cuda doesn't support bool

    # Create decoder_input_ids by shifting the input_ids right
    decoder_input_ids = input_ids.clone()
    decoder_input_ids[:, :-1] = input_ids[:, 1:]
    decoder_input_ids[:, -1] = processor.tokenizer.eos_token_id

    # Forward pass
    with torch.no_grad():
        logits = model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
        ).logits

    batch_size, seq_len, vocab_size = logits.shape
    max_length = (end_idx - start_idx).max().item()

    # init tensors with -100 padding to max length for efficient batched CE loss calculation
    padded_logits = torch.full((batch_size, max_length, vocab_size), -100.0, device=device)
    padded_input_ids = torch.full((batch_size, max_length), -100, device=device)

    # Fill in the relevant portions
    for i in range(batch_size):
        length = end_idx[i] - start_idx
        padded_logits[i, :length] = logits[i, start_idx-2:end_idx[i]-2] # account for decoder offset
        padded_input_ids[i, :length] = input_ids[i, start_idx:end_idx[i]]

    # calculate per-token loss and perplexity
    loss = F.cross_entropy(padded_logits.view(-1, vocab_size), 
                           padded_input_ids.view(-1), 
                           ignore_index=-100, 
                           reduction='none')
    loss = loss.view(batch_size, -1)
    perplexity = torch.exp(loss.sum(dim=1) / (end_idx - start_idx).float())
    return perplexity.tolist()

if __name__ == "__main__":

    main()

    
    # # FIXME: known issue that batching influences scores within the batch for some reason. One factor is that padding to longest in batch influences decoding scores
    # # this is because the attention mask that usually ignores padding tokens is not used in generate, which causes the model to attend over the padding tokens
    # # For example, if you set padding="max_length", the resulting scores for both true and false ground truth are relatively low scores.
     
    # ## unit test: batched inference
    # image_path_1 = "data/sampled_datacomp/sampled_datacomp_images/000000000002.jpg"
    # caption_1 = "The image features a Mercedes-Benz G-Class SUV, portrayed in a professional studio setting. The vehicle sports a metallic gray finish, with a robust and boxy design that emphasizes its luxury and off-road capabilities. It features distinctive elements such as a bold front grille adorned with the Mercedes-Benz logo, round LED headlights, and a prominent front bumper. The SUV's rugged appeal is enhanced by its pronounced wheel arches housing large, stylish alloy wheels. The side profile reveals sturdy running boards and a high ground clearance, typical of a vehicle designed for both urban and off-road use. The overall aesthetic combines a traditional SUV silhouette with modern, high-end details, presenting a vehicle that exudes both luxury and utility." # positive test case
    # image_path_2 = "data/sampled_datacomp/sampled_datacomp_images/000000000007.jpg"
    # caption_2 = "bad example what is this" # negative test case
    # image_path_3 = "data/sampled_datacomp/sampled_datacomp_images/000000004230.jpg"
    # caption_3 = "the highlighted vested woman is on the left of the man with a blue shirt" # negative test case

    # repeated_caption_1 = caption_3

    # print("Testing batched inference")
    # images, captions = [image_path_1, image_path_2, image_path_3], [caption_1, caption_2, caption_3]
    # scores = forward(images, captions)
    # print(scores)

    # print("Testing 3x single forward inference")
    # for image, caption in zip(images, captions):
    #     score = forward([image], [caption])
    #     print(score)
