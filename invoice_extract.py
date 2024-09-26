"""

Python script to extract text from invoices using ML model.

Input:      Invoices of PDF filetype.
Output:     TXT files of extracted texts.


ML Model: Sebabrata/dof-invoice-1
Link: https://huggingface.co/Sebabrata/dof-invoice-1
License: MIT


How To Use Script:
    1. Define the variable input_invoice_folder, path to the folder where invoices were stored in.
    2. Run script.
    

Last Edit: 25/09/2024

"""


# ========================================
# User-defined Parameters

base_folder_path = r"C:\Users\Administrator\Desktop\Docs"       # folder where the whole AI text extraction process takes place
source_folder_name = "2AI"                                      # name of the folder where inputs are stored
destination_folder_name = "8AI"                                 # name of the folder where outputs are stored
temporary_folder_name = "System"                                # name of the folder where files are stored temporarily
temporary_storage_folder_name = "wip_files"                     # folder to store files for extraction
converted_invoice_folder_name = "invoice_png"                   # folder used to store converted PDFs images temporarily, will be deleted after completion of task
# ========================================


# ========================================
# Imports

import os
import shutil
import torch
import re
from PIL import Image
from transformers import DonutProcessor, VisionEncoderDecoderModel
from pdf2image import convert_from_path
# ========================================


# ========================================
# Parameters

source_folder_path = os.path.join(base_folder_path, source_folder_name)                             # folder where all the inputs are stored in
destination_folder_path = os.path.join(base_folder_path, destination_folder_name)                   # folder where all created TXT files and original PDFs are stored in
temporary_folder_path = os.path.join(base_folder_path, temporary_folder_name)                       # folder where the AI model and all temporarily created folders are stored in
wip_folder_path = os.path.join(temporary_folder_path, temporary_storage_folder_name)                # folder to temporarily store files
converted_invoice_folder_path = os.path.join(temporary_folder_path, converted_invoice_folder_name)  # folder to temporarily store converted images

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")                             # use cuda if available, else cpu
processor = DonutProcessor.from_pretrained(r".\models\01_dof-invoice-1")                            # select processor from "models" folder
model = VisionEncoderDecoderModel.from_pretrained(r".\models\01_dof-invoice-1")                     # select VisionEncoderDecoderModel from "models" folder
model.to(device)                                                                                    # link model to device
# ========================================


# ========================================
# Basic Functions

# Create folder
def create_folder(folder_path):
    try:
        os.mkdir(folder_path)
    except FileExistsError:
        pass


# Check if folder exists
def check_main_folders():
    if os.path.exists(source_folder_path) is False:
        raise Exception("No 'Source_Folder' found. Please check if you had created the folder 'Source_Folder' under " +
                        base_folder_path + " and placed the PDFs you wished to extract in " + source_folder_path + ".")
    else:
        if os.path.exists(destination_folder_path) is False:
            create_folder(destination_folder_path)
        else:
            pass

        if os.path.exists(temporary_folder_path) is False:
            create_folder(temporary_folder_path)
        else:
            pass


# Replicate directory from source to destination
def replicate_dir(source_path, destination_path):
    for (root, dirs, files) in os.walk(source_path, topdown = True):
        new_empty_folder_path = root.replace(source_path, destination_path)

        try:
            os.makedirs(new_empty_folder_path)
        except FileExistsError:
            pass


# Move files from source to destination
def move_files(source_path, destination_path, converted_image_bool = False):
    for (root, dirs, files) in os.walk(source_path, topdown = True):
        if len(files) != 0 and converted_image_bool == False:
            for f in files:
                if f.endswith(".txt") or f.endswith(".pdf"):
                    f_path = os.path.join(root, f)
                    new_f_path = f_path.replace(source_path, destination_path)
                    os.rename(f_path, new_f_path)
                else:
                    pass
        elif len(files) != 0 and converted_image_bool == True:
            for f in files:
                if f.endswith(".png"):
                    f_path = os.path.join(root, f)
                    new_f_path = f_path.replace(root, destination_path)
                    os.rename(f_path, new_f_path)
                else:
                    pass
        else:
            pass


# Check if the directories in source are duplicated to destination
# If yes, delete corresponding directories in source
def check_and_delete(source_path, destination_path):
    for (root, dirs, files) in os.walk(source_path, topdown = False):
        d_root = root.replace(source_path, destination_path)
        count = 0

        if len(dirs) == 0 and len(files) == 0 and os.path.exists(d_root):
            pass
        elif len(dirs) != 0:
            for d in dirs:
                if os.path.exists(os.path.join(d_root, d)):
                    pass
                else:
                    count += 1
        elif len(files) != 0:
            for f in files:
                if os.path.exists(os.path.join(d_root, f)):
                    pass
                else:
                    count += 1
        else:
            pass

    if count == 0:
        for i in os.listdir(source_path):
            shutil.rmtree(os.path.join(source_path, i))
    else:
        print("Deleting all subdirectories in %s has failed." %source_path)
        print("Some folder(s) are not empty, hence, not deleted.")
        print("Please check if %s and %s has duplicated file(s) or subdirectories." %source_path %destination_path)
    

# Delete directories in temporary folder
def delete_temp():
    for (root, dirs, files) in os.walk(wip_folder_path, topdown = True):
        if len(dirs) == 0 and len(files) == 0:
            shutil.rmtree(root)
        else:
            pass

    for (root, dirs, files) in os.walk(converted_invoice_folder_path, topdown = False):
        if len(files) != 0:
            for f in files:
                os.remove(os.path.join(root, f))
        else:
            pass
    
    for (root, dirs, files) in os.walk(converted_invoice_folder_path, topdown = True):
        if len(dirs) == 0 and len(files) == 0:
            shutil.rmtree(root)
        else:
            pass


# Move files and folders from source to destination
def move(source_path, destination_path, mode = 0):
    # for moving directories and files from source to wip folder
    # for moving extracted txt files and original pdfs from wip folder to destination folder
    if mode == 0:
        replicate_dir(source_path, destination_path)
        move_files(source_path, destination_path, False)
        check_and_delete(source_path, destination_path)
    # for moving converted image files from wip folder to converted image folder
    elif mode == 1:
        move_files(source_path, destination_path, True)
    else:
        pass


# Convert PDF to PNG
def convert_pdf(file_path):
    img_list = []

    if file_path.endswith(".pdf"):
        pdf_img = convert_from_path(file_path)
        file_path_stripped = file_path.rstrip(".pdf")

        for i in range(len(pdf_img)):
            img_path = file_path_stripped + "_page_" + str(i + 1) + ".png"
            pdf_img[i].save(img_path, "PNG")
            img_list.insert(0, img_path)
    elif file_path.endswith(".jpg") or file_path.endswith(".jpeg") or file_path.endswith(".png"):
        img_list.append(file_path)
    else:
        pass

    return img_list


# Load and preprocess images
def load_and_preprocess_image(img_path):
    image = Image.open(img_path).convert("RGB")
    pixel_values = processor(image, return_tensors = "pt").pixel_values

    return pixel_values


# Extract texts from images
def generate_text_from_image(pixel_values):
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


# Filter texts extracted
def filter_function(pair):
    wanted_keys = ["supplier_name", "supplier_address", "receiver_name", "receiver_address",
                   "invoice_id", "invoice_date", "due_date", "payment_terms",
                   "tax_amount", "sub_total", "total"]
    
    key, value = pair

    if key not in wanted_keys:
        return False
    elif value == "":
        return False
    else:
        return True


# Merge filtered texts
def merge_outputs(output_list):
    if len(output_list) > 1:
        i = 0

        while i in range(len(output_list) - 1):
            output_list[i + 1] = output_list [i] | output_list[i + 1]
            i += 1

        output = output_list[len(output_list) - 1]
    elif len(output_list) == 1:
        output = output_list[0]
    else:
        raise Exception("No output detected.")
    
    return output


# Create TXT files and log merged texts
def log_text(extracted_text, file_root, file_name):
    file_name = file_name.rstrip(".pdf")
    file_name = file_name.rstrip(".png")
    file_name = file_name.rstrip(".jpg")
    output_file_name = os.path.join(file_root, file_name) + ".txt"

    with open(output_file_name, "w", encoding = "utf-8") as outfile:
        for key, value in extracted_text.items():
            outfile.write("%s: %s\n" % (key, value))
        
    print(file_name + " extracted.")


# Integrated function to extract and log texts
def extract_and_log(img_list, file_root, file_name):
    extracted_list = []

    for img_path in img_list:
        pixel_values = load_and_preprocess_image(img_path)
        extracted_text = generate_text_from_image(pixel_values)

        if type(extracted_text) == dict:
            extracted_text = dict(filter(filter_function, extracted_text.items()))
        elif type(extracted_text) == list:
            extracted_text = dict(filter(filter_function, extracted_text[0].items()))
        else:
            raise Exception("Extracted text is not either of dictionary or list type.")
        
        extracted_list.append(extracted_text)
    
    final_extracted_text = merge_outputs(extracted_list)
    log_text(final_extracted_text, file_root, file_name)
# ========================================


# ========================================
# Main Function

def extract_invoice(folder_path):
    check_main_folders()
    create_folder(wip_folder_path)
    create_folder(converted_invoice_folder_path)
    move(source_folder_path, wip_folder_path, 0)
    total_extracted = 0

    print("\n")
    print("========== Starting to extract texts from PDFs ==========")
    print("\n")

    for (root, dirs, files) in os.walk(wip_folder_path, topdown = True):
       if len(files) != 0:
           for f in files:
               f_path = os.path.join(root, f)
               f_img_list = convert_pdf(f_path)

               if len(f_img_list) == 0:
                   pass
               else:
                   extract_and_log(f_img_list, root, f)
                   total_extracted += 1
       else:
           pass
    
    move(wip_folder_path, converted_invoice_folder_path, 1)
    move(wip_folder_path, destination_folder_path, 0)
    delete_temp()
    
    print("\n")
    print("Extraction completed. Total " + str(total_extracted) + " file(s) extracted.")
    print("Please head to " + destination_folder_path + " for the extracted_text_files.")
    print("========== END ==========")
# ========================================


# ========================================
# Execution

extract_invoice(base_folder_path)
# ========================================