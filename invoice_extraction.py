"""

Python script to extract text from invoices using ML model.

Input:      Invoices of PDF, JPEG or PNG filetype.
Output:     TXT files of extracted texts.


ML Model: Sebabrata/dof-invoice-1
Link: https://huggingface.co/Sebabrata/dof-invoice-1
License: MIT


How To Use Script:
    1. Define the variable input_invoice_folder, path to the folder where invoices were stored in.
    2. Run script.
    

Last Edit: 21/03/2024

"""


# ========================================
# User-defined Parameters

input_invoice_folder = r"C:\Users\Evelyn\Desktop\input_invoice"     # path to input folder
processed_invoice_folder = r"processed_invoice"                     # folder to store processed invoices
output_text_folder = r"output_text"                                 # folder to output folder
converted_invoice_folder = r"converted_invoice"                     # folder to converted invoices
# ========================================


# ========================================
# Imports

import os
import torch
import re
from PIL import Image
from transformers import DonutProcessor, VisionEncoderDecoderModel
from pdf2image import convert_from_path
# ========================================


# ========================================
# Parameters

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
processor = DonutProcessor.from_pretrained("./models/01_dof-invoice-1")
model = VisionEncoderDecoderModel.from_pretrained("./models/01_dof-invoice-1")
model.to(device)
# ========================================


# ========================================
# Functions

def create_folder(folder_name):
    try:
        os.mkdir(input_invoice_folder + "\\" + folder_name)
    except FileExistsError:
        pass


def convert_pdf(file_path):
    img_list = []
    
    if file_path.endswith(".pdf"):
        pdf_img = convert_from_path(file_path)
        file_path_stripped = file_path.rstrip(".pdf")
        
        for i in range(len(pdf_img)):
            img_path = file_path_stripped + "_page_" + str(i+1) + ".png"
            pdf_img[i].save(img_path, "PNG")
            img_list.insert(0, img_path)
    elif file_path.endswith(".jpg") or file_path.endswith(".jpeg") or file_path.endswith(".png"):
        img_list.append(file_path)
    else:
        pass
    
    return img_list


def load_and_preprocess_image(img_path: str, processor):
    image = Image.open(img_path).convert("RGB")
    pixel_values = processor(image, return_tensors = "pt").pixel_values
    
    return pixel_values


def generate_text_from_image(model, pixel_values, processor, device):
    pixel_values = pixel_values.to(device)
    model.eval()
    
    with torch.no_grad():
        task_prompt = "<s_cord-v2>"
        decoder_input_ids = processor.tokenizer(task_prompt, add_special_tokens = False, return_tensors = "pt").input_ids
        decoder_input_ids = decoder_input_ids.to(device)
        
        generated_outputs = model.generate(
            pixel_values,
            decoder_input_ids = decoder_input_ids,
            max_length = model.decoder.config.max_position_embeddings,
            pad_token_id = processor.tokenizer.pad_token_id,
            eos_token_id = processor.tokenizer.eos_token_id,
            early_stopping = True,
            bad_words_ids = [[processor.tokenizer.unk_token_id]],
            return_dict_in_generate = True
        )
        
    decoded_text = processor.batch_decode(generated_outputs.sequences)[0]
    decoded_text = decoded_text.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
    decoded_text = re.sub(r"<.*?>", "", decoded_text, count = 1).strip()
    decoded_text = processor.token2json(decoded_text)
    
    return decoded_text


def filter_function(pair):
    wanted_keys = ["supplier_name", "supplier_address", "receiver_name", "receiver_address",
                   "invoice_id", "invoice_date",
                   "tax_amount", "sub_total", "total"]
    key, value = pair
    
    if key not in wanted_keys:
        return False
    elif value == "":
        return False
    else:
        return True
    

def merge_outputs(output_list):
    if len(output_list) > 1:
        i = 0
        
        while i in range(len(output_list) - 1):
            output_list[i + 1] = output_list[i] | output_list[i + 1]
            i += 1
            
        output = output_list[len(output_list) - 1]
    elif len(output_list) == 1:
        output = output_list[0]
    else:
        raise Exception("No output detected.")
    
    return output


def log_text(extracted_text, file_name):
    file_name = file_name.strip(".pdf")
    file_name = file_name.strip(".png")
    file_name = file_name.strip(".jpg")
    output_file_name = input_invoice_folder + "\\" + output_text_folder + "\\" + file_name + ".txt"
    
    with open(output_file_name, "w", encoding="utf-8") as outfile:
        for key, value in extracted_text.items():  
            outfile.write('%s:%s\n' % (key, value))
    
    print(file_name + " extracted.")


def move_processed_file(source_path, destination_folder, file_name):
    file_name = file_name.strip(input_invoice_folder)
    destination_path = input_invoice_folder + "\\" + destination_folder + "\\" + file_name
    os.rename(source_path, destination_path)


def extract_invoice(invoice_folder):
    create_folder(processed_invoice_folder)
    create_folder(output_text_folder)
    create_folder(converted_invoice_folder)
    
    file_list = os.listdir(invoice_folder)
    
    for i in range(len(file_list)):
        file_path = invoice_folder + "\\" + file_list[i]
        image_list = convert_pdf(file_path)
        
        if len(image_list) == 0 and i == 0:
            raise Exception("No files of suitable types to be extracted. Please make sure the input file type is either .pdf, .png, .jpeg or .jpg")
        elif len(image_list) == 0 and i != 0:
            pass
        else:
            extracted_list = []
            
            for j in image_list:
                pixel_values = load_and_preprocess_image(j, processor)
                extracted_text = generate_text_from_image(model, pixel_values, processor, device)
                
                if type(extracted_text) == dict:
                    extracted_text = dict(filter(filter_function, extracted_text.items()))
                elif type(extracted_text) == list:
                    extracted_text = dict(filter(filter_function, extracted_text[0].items()))
                else:
                    raise Exception("Extracted text is not either of dictionary or list types.")
                    
                extracted_list.append(extracted_text)
                move_processed_file(j, converted_invoice_folder, j)
            
            final_extracted_text = merge_outputs(extracted_list)
            log_text(final_extracted_text, file_list[i])
            
            move_processed_file(file_path, processed_invoice_folder, file_list[i])
        
    print("Extraction completed. Please head to " + input_invoice_folder + "\\" + output_text_folder + " for the extracted text files.")
# ========================================


# ========================================
# Execution

extract_invoice(input_invoice_folder)
# ========================================