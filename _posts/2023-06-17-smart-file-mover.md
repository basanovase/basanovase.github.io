
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
import requests
import json

# Setup logging
logging.basicConfig(level=logging.INFO)



def classify_file(filename, prompt_template):
    """
    Classify file or folder using GPT-4.
    """
    openai.api_key = keyring.get_password('openai','burping')

    prompt = prompt_template.format(filename=filename)
    response = openai.Completion.create(engine="text-davinci-003", prompt=prompt, max_tokens=5)

    # Use Google Custom Search API for more context
    google_search_api_key = keyring.get_password('google','burping') # Replace with your API key
    google_cx = keyring.get_password('google_cx','burping') # Replace with your Custom Search JSON API cx


    url = f"https://www.googleapis.com/customsearch/v1?key={google_search_api_key}&cx={google_cx}&q={filename}"
    response = requests.get(url)
    search_results = json.loads(response.text)

    # Use the snippet from the first search result as additional context
    if 'items' in search_results and len(search_results['items']) > 0:
        logging.info(f"Hmm, not sure about that one, googling it!")
        context = search_results['items'][0]['snippet']
        prompt += f"\nAdditional context: {context}"
        logging.info(f"I found this info: {context}")
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
        "ebook_pdf_technical": "eBooks and PDFs/Technical",
        "ebook_pdf_novels": "eBooks and PDFs/Novels",
        "ebook_pdf_research": "eBooks and PDFs/Research Papers",
        "ebook_pdf_educational": "eBooks and PDFs/Educational",
        "ebook_pdf_business": "eBooks and PDFs/Business",
        "job_descriptions": "Useful Documents/Job Descriptions",
        "business_process": "Useful Documents/Business Process",
        "financial_statements": "Useful Documents/Financial Statements",
        "contracts_agreements": "Useful Documents/Contracts and Agreements",
        "meeting_notes": "Useful Documents/Meeting Notes",
        "csv": "Data",
        "other": "Others"
    }

    # Template for GPT-4 prompt
    prompt_template = """
    Given the filename '{filename}', which category does it best fit into: 
    application, audio, image, message, text, video, zip, code, 
    ebook_pdf_technical, ebook_pdf_novels, ebook_pdf_research, ebook_pdf_educational, ebook_pdf_business, 
    job_descriptions, business_process, financial_statements, contracts_agreements, meeting_notes, csv, or other?
    Use your knowledge of file types in a Downloads folder. Take into account their extensions or name."""

    # Iterate over files and directories in Downloads directory
    for filename in os.listdir(path_to_downloads):
        file_path = os.path.join(path_to_downloads, filename)

        if filename in folder_map.values():
            logging.info(f"Skipping bucket: {filename}")
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
        os.makedirs(new_folder_path, exist_ok=True)
        logging.info(f"Created folder: {new_folder_path}")

        # Move file or directory to the new folder
        new_file_path = os.path.join(new_folder_path, filename)
        shutil.move(file_path, new_file_path)
        logging.info(f"Moved: {file_path} to {new_file_path}")

    logging.info("Downloads directory cleanup completed successfully.")

# Usage
clean_downloads("/Users/flynnmclean/Downloads")




{% endhighlight %}


