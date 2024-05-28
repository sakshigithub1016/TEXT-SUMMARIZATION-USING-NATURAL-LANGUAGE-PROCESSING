import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from collections import Counter
from heapq import nlargest
from transformers import pipeline
from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize
from rouge import Rouge
import tkinter as tk
from tkinter import scrolledtext
from tkinter import messagebox

# Function to summarize text using spaCy
def summarize_text_spacy(text, num_sentences):
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(text)
    tokens = [token.text.lower() for token in doc
              if not token.is_stop and not token.is_punct and token.text != '\n']
    
    word_freq = Counter(tokens)
    if not word_freq:
        return "No words found in the text."
    
    max_freq = max(word_freq.values())
    for word in word_freq.keys():
        word_freq[word] = word_freq[word]/max_freq
    
    sent_token = [sent.text for sent in doc.sents]
    
    sent_score = {}
    for sent in sent_token:
        for word in sent.split():
            if word.lower() in word_freq.keys():
                if sent not in sent_score.keys():
                    sent_score[sent] = word_freq[word]
                else:
                    sent_score[sent] += word_freq[word]
    
    summarized_sentences = nlargest(num_sentences, sent_score, key=sent_score.get)
    summarized_text = " ".join(summarized_sentences)
    return summarized_text

# Function to summarize text using T5 transformer
def summarize_text_t5(text):
    summarizer = pipeline("summarization", model="t5-base", tokenizer="t5-base", framework="pt")
    summary = summarizer(text, max_length=100, min_length=10, do_sample=False)
    return summary[0]['summary_text']

# Function to compute BLEU score
def compute_bleu_score(reference_summary, generated_summary):
    reference_tokens = word_tokenize(reference_summary.lower())
    generated_tokens = word_tokenize(generated_summary.lower())
    bleu_score = sentence_bleu([reference_tokens], generated_tokens)
    return bleu_score

# Function to compute ROUGE scores
def compute_rouge_scores(reference_summary, generated_summary):
    rouge = Rouge()
    scores = rouge.get_scores(generated_summary, reference_summary)
    return scores

# Function to summarize text using the selected method and display the results
def summarize_and_display():
    input_text = input_text_box.get("1.0", tk.END)
    num_sentences = num_sentences_slider.get()
    method = method_var.get()
    
    if method == "spaCy":
        summarized_text = summarize_text_spacy(input_text, num_sentences)
    elif method == "T5":
        summarized_text = summarize_text_t5(input_text)
    else:
        messagebox.showerror("Error", "Invalid summarization method.")
        return
    
    output_text_box.delete("1.0", tk.END)
    output_text_box.insert(tk.END, summarized_text)
    
    # Compute BLEU score and ROUGE scores for T5 method
    if method == "T5":
        bleu_score = compute_bleu_score(reference_summary, summarized_text)
        rouge_scores = compute_rouge_scores(reference_summary, summarized_text)
        messagebox.showinfo("Evaluation Metrics", f"BLEU Score: {bleu_score}\nROUGE Scores: {rouge_scores}")

# Create the main window
root = tk.Tk()
root.title("Text Summarization GUI")

# Create input text box
input_text_box = scrolledtext.ScrolledText(root, width=50, height=10, wrap=tk.WORD)
input_text_box.grid(row=0, column=0, padx=10, pady=10)

# Create a label and slider for selecting number of sentences
num_sentences_label = tk.Label(root, text="Select number of sentences:")
num_sentences_label.grid(row=1, column=0, padx=10, pady=5)

num_sentences_slider = tk.Scale(root, from_=1, to=10, orient=tk.HORIZONTAL)
num_sentences_slider.set(3)
num_sentences_slider.grid(row=1, column=1, padx=10, pady=5)

# Create a radio button for selecting summarization method
method_var = tk.StringVar(value="spaCy")

method_label = tk.Label(root, text="Select summarization method:")
method_label.grid(row=2, column=0, padx=10, pady=5)

spaCy_radio = tk.Radiobutton(root, text="spaCy", variable=method_var, value="spaCy")
spaCy_radio.grid(row=2, column=1, padx=10, pady=5, sticky="w")

T5_radio = tk.Radiobutton(root, text="T5", variable=method_var, value="T5")
T5_radio.grid(row=3, column=1, padx=10, pady=5, sticky="w")

# Create a button to summarize text
summarize_button = tk.Button(root, text="Summarize", command=summarize_and_display)
summarize_button.grid(row=4, column=0, columnspan=2, padx=10, pady=10)

# Create output text box
output_text_box = scrolledtext.ScrolledText(root, width=50, height=10, wrap=tk.WORD)
output_text_box.grid(row=5, column=0, padx=10, pady=10, columnspan=2)

# Reference summary for evaluation metrics
reference_summary = "In a world often dominated by negativity, it's crucial to remember the potency of kindness and compassion. Every act, no matter how small, can ignite positivity and foster a chain reaction of goodwill. When individuals and communities unite, they wield remarkable power to drive meaningful change, enriching lives and inspiring hope. Let's harness our innate ability to make a difference and cultivate a culture of empathy, creating a brighter future for all through collective kindness."

# Run the GUI
root.mainloop()
