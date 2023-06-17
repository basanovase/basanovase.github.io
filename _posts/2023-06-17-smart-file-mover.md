
---
title:  "Watching birds on the beach"
mathjax: true
layout: post
categories: media
---




A fun little experiment with using ChatGPT to aid with file classification


{% highlight python %}

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 17 20:08:17 2023

@author: flynnmclean
"""

import os
import shutil
import mimetypes
import openai
import keyring
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)

def classify_file(filename, prompt_template):
    """
    Classify file or folder using GPT-4.
    """
    openai.api_key = keyring.get_password('openai','burping')

    prompt = prompt_template.format(filename=filename)
    response = openai.Completion.create(engine="text-davinci-003", prompt=prompt, max_tokens=5)

    return response.choices[0].text.strip()

def clean_downloads(path_to_downloads):
    """
    Clean up the downloads folder by categorizing files and directories 
    into sub-folders based on their types and GPT-4 classification.
    """

    # Dictionary for mapping file types and classifications to folders
    folder_map = {
        "application": "Applications",
        "audio": "Audios",
        "image": "Images",
        "message": "Messages",
        "text": "Texts",
        "video": "Videos",
        "zip": "Personal Documentation",
        "code": "Code",
        "ebook_pdf": "eBooks and PDFs",
        "csv": "Data",
        "other": "Others"
    }

    # Template for GPT-4 prompt
    prompt_template = "Given the filename '{filename}', which category does it best fit into: application, audio, image, message, text, video, zip, code, ebook_pdf, csv, or other? Use your knowledge of file types in a Downloads folder. Take into account their extensions or name."

    # Iterate over files and directories in Downloads directory
    for filename in os.listdir(path_to_downloads):
        file_path = os.path.join(path_to_downloads, filename)

        # Skip 'bucket' directories
        if filename in folder_map.values():
            logging.info(f"Skipping bucket directory: {filename}")
            continue

        # Guess file type if it's a file
        if os.path.isfile(file_path):
            file_type, _ = mimetypes.guess_type(filename)
            if file_type is None or file_type.split('/')[0] not in folder_map:
                file_type = 'other'
            else:
                file_type = file_type.split('/')[0]
        else:
            file_type = 'other'

        # Classify file or directory using GPT-4
        classification = classify_file(filename, prompt_template)

        # Use the GPT-4 classification if it's in the folder map, otherwise use the guessed file type
        folder = folder_map.get(classification, folder_map[file_type])

        # Create new folder if not exist
        new_folder_path = os.path.join(path_to_downloads, folder)
        if not os.path.exists(new_folder_path):
            os.mkdir(new_folder_path)
            logging.info(f"Created folder: {new_folder_path}")

        # Move file or directory to the new folder
        new_file_path = os.path.join(new_folder_path, filename)
        shutil.move(file_path, new_file_path)
        logging.info(f"Moved: {file_path} to {new_file_path}")

    logging.info("Downloads directory cleanup completed successfully.")

# Usage
clean_downloads("/Users/flynnmclean/Downloads")

{% endhighlight %}


