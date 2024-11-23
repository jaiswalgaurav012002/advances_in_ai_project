import subprocess
import sys
import os

# Function to install packages programmatically
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Install necessary libraries
install('langdetect')
install('textstat')
install('wordcloud')
install('nltk')

# Import necessary libraries
import pandas as pd
import numpy as np
import re
import random
from nltk.corpus import wordnet
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer
from textstat import flesch_reading_ease
from sklearn.metrics import classification_report, confusion_matrix

# Download NLTK data
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt')

# Define the base path
base_path = '/storage/research/data/Few_shot/Ai_project/'

# Ensure the output directory exists
os.makedirs(base_path, exist_ok=True)

# Load the dataset
file_path = os.path.join(base_path, 'english_data.csv')
df = pd.read_csv(file_path, header=None, names=['raw_data'])

# Process the dataset
df[['sentence', 'type_of_speech']] = df['raw_data'].str.rsplit('\t', n=1, expand=True)
df = df.drop(columns=['raw_data'])
df['sentence'] = df['sentence'].str.strip()
df['type_of_speech'] = df['type_of_speech'].astype(str).str.strip().replace('nan', '')

# Define keywords indicating type_of_speech labels
keywords = ["Hope_speech", "Non_hope_speech"]

# Function to check and correct misaligned labels within the sentence
def extract_label(row):
    for keyword in keywords:
        if keyword in row['sentence']:
            row['type_of_speech'] = keyword
            row['sentence'] = row['sentence'].replace(f"\t{keyword}", "").strip()
    return row

# Apply the function to clean up misaligned labels
df = df.apply(extract_label, axis=1)

# Function to remove special characters but keep punctuation
def remove_special_characters(text):
    return re.sub(r'[^A-Za-z0-9\s.,!?\'"-]', '', text)

df['sentence'] = df['sentence'].apply(remove_special_characters)

# Detect and filter out non-English sentences
from langdetect import detect, DetectorFactory
DetectorFactory.seed = 0

def is_english(text):
    try:
        return detect(text) == 'en'
    except:
        return False

df = df[df['sentence'].apply(is_english)]

# Save the cleaned dataset
output_path = os.path.join(base_path, 'cleaned_english_data_english_only.csv')
df.to_csv(output_path, index=False)
print(f"Cleaned dataset saved to {output_path}")

# Perform Exploratory Data Analysis (EDA)

# Dataset dimensions
print("Dataset dimensions:", df.shape)

# Count number of instances for each type_of_speech
speech_type_distribution = df['type_of_speech'].value_counts()
print("\nDistribution of type_of_speech:\n", speech_type_distribution)

# Calculate word and character count statistics
df['word_count'] = df['sentence'].str.split().str.len()
df['char_count'] = df['sentence'].str.len()

word_count_stats = {
    "Min Word Count": df['word_count'].min(),
    "Max Word Count": df['word_count'].max(),
    "Mean Word Count": df['word_count'].mean(),
    "Median Word Count": df['word_count'].median()
}
print("\nWord Count Statistics:\n", word_count_stats)

char_count_stats = {
    "Min Char Count": df['char_count'].min(),
    "Max Char Count": df['char_count'].max(),
    "Mean Char Count": df['char_count'].mean(),
    "Median Char Count": df['char_count'].median()
}
print("\nCharacter Count Statistics:\n", char_count_stats)

# Save visualizations
visualizations_path = os.path.join(base_path, 'visualizations')
os.makedirs(visualizations_path, exist_ok=True)

# Histogram of sentence lengths (in words)
plt.figure(figsize=(10, 6))
plt.hist(df['word_count'], bins=30, edgecolor='k', alpha=0.7)
plt.title("Histogram of Sentence Lengths (Words)")
plt.xlabel("Number of Words")
plt.ylabel("Frequency")
plt.savefig(os.path.join(visualizations_path, 'histogram_sentence_lengths.png'))
plt.show()

# Bar chart for distribution of type_of_speech
plt.figure(figsize=(8, 5))
speech_type_distribution.plot(kind='bar', color='skyblue', edgecolor='k')
plt.title("Distribution of Type of Speech")
plt.xlabel("Type of Speech")
plt.ylabel("Count")
plt.xticks(rotation=0)
plt.savefig(os.path.join(visualizations_path, 'speech_type_distribution.png'))
plt.show()

# Word Cloud for Hope Speech
hope_text = " ".join(df[df['type_of_speech'] == 'Hope_speech']['sentence'])
wordcloud_hope = WordCloud(width=800, height=400, background_color="white").generate(hope_text)
plt.figure(figsize=(10, 6))
plt.imshow(wordcloud_hope, interpolation='bilinear')
plt.axis("off")
plt.title("Word Cloud for Hope Speech")
plt.savefig(os.path.join(visualizations_path, 'wordcloud_hope_speech.png'))
plt.show()

# Word Cloud for Non-Hope Speech
non_hope_text = " ".join(df[df['type_of_speech'] == 'Non_hope_speech']['sentence'])
wordcloud_non_hope = WordCloud(width=800, height=400, background_color="white").generate(non_hope_text)
plt.figure(figsize=(10, 6))
plt.imshow(wordcloud_non_hope, interpolation='bilinear')
plt.axis("off")
plt.title("Word Cloud for Non-Hope Speech")
plt.savefig(os.path.join(visualizations_path, 'wordcloud_non_hope_speech.png'))
plt.show()

print(f"All visualizations have been saved to {visualizations_path}")















import pandas as pd

# Load the dataset
file_path = '/storage/research/data/Few_shot/Ai_project/english_data.csv'  # Updated path
df = pd.read_csv(file_path, header=None, names=['raw_data'])

# Split 'raw_data' into 'sentence' and 'type_of_speech' based on the last tab character
df[['sentence', 'type_of_speech']] = df['raw_data'].str.rsplit('\t', n=1, expand=True)

# Drop the 'raw_data' column as it’s no longer needed
df = df.drop(columns=['raw_data'])

# Clean up whitespace
df['sentence'] = df['sentence'].str.strip()
df['type_of_speech'] = df['type_of_speech'].astype(str).str.strip().replace('nan', '')

# Further process to clean up any rows where 'type_of_speech' label is embedded within the sentence itself

# Define keywords indicating type_of_speech labels
keywords = ["Hope_speech", "Non_hope_speech"]

# Function to check and correct misaligned labels within the sentence
def extract_label(row):
    for keyword in keywords:
        if keyword in row['sentence']:
            row['type_of_speech'] = keyword
            row['sentence'] = row['sentence'].replace(f"\t{keyword}", "").strip()
    return row

# Apply the function to clean up misaligned labels
df = df.apply(extract_label, axis=1)

# Display the first few rows to verify the cleaned data
print(df.head(10))

# Save the cleaned dataset to a new CSV file
output_path = '/storage/research/data/Few_shot/Ai_project/modified_english_data.csv'  # Updated path
df.to_csv(output_path, index=False)
import re

# Basic exploratory data analysis

# Size and shape of the dataset
data_shape = df.shape
num_data_points = len(df)

# Function to clean text and count unique words
def count_distinct_words(text_series):
    # Join all text data into one large string
    all_text = ' '.join(text_series)
    # Remove special characters and punctuation using regex, then split into words
    words = re.findall(r'\b\w+\b', all_text.lower())
    # Return the number of unique words
    return len(set(words))

# Count number of distinct words in the 'sentence' column
num_distinct_words = count_distinct_words(df['sentence'])

# Display the results
data_shape, num_data_points, num_distinct_words

import pandas as pd
import re
import os

# Load the dataset
file_path = '/storage/research/data/Few_shot/Ai_project/english_hope_train.csv'  # Updated path
df = pd.read_csv(file_path, header=None, names=['raw_data'])

# Split 'raw_data' into 'sentence' and 'type_of_speech' based on the last tab character
df[['sentence', 'type_of_speech']] = df['raw_data'].str.rsplit('\t', n=1, expand=True)

# Drop the 'raw_data' column as it’s no longer needed
df = df.drop(columns=['raw_data'])

# Clean up whitespace
df['sentence'] = df['sentence'].str.strip()
df['type_of_speech'] = df['type_of_speech'].astype(str).str.strip().replace('nan', '')

# Further process to clean up any rows where 'type_of_speech' label is embedded within the sentence itself

# Define keywords indicating type_of_speech labels
keywords = ["Hope_speech", "Non_hope_speech"]

# Function to check and correct misaligned labels within the sentence
def extract_label(row):
    for keyword in keywords:
        if keyword in row['sentence']:
            row['type_of_speech'] = keyword
            row['sentence'] = row['sentence'].replace(f"\t{keyword}", "").strip()
    return row

# Apply the function to clean up misaligned labels
df = df.apply(extract_label, axis=1)

# Function to remove special characters but keep punctuation
def remove_special_characters(text):
    # Remove all characters that are not letters, numbers, spaces, or punctuation
    return re.sub(r'[^A-Za-z0-9\s.,!?\'"-]', '', text)

# Apply the function to the 'sentence' column
df['sentence'] = df['sentence'].apply(remove_special_characters)

# Ensure the directory exists or create it if running locally
os.makedirs('/storage/research/data/Few_shot/Ai_project', exist_ok=True)

# Save the modified dataset to a new CSV file
output_path = '/storage/research/data/Few_shot/Ai_project/cleaned_english_data.csv'  # Updated path
df.to_csv(output_path, index=False)

# Print the location of the saved file
print(f"Modified dataset saved to {output_path}")


import pandas as pd
import re
from langdetect import detect, DetectorFactory

DetectorFactory.seed = 0  # Ensures consistent results in language detection

# Load the dataset
file_path = '/storage/research/data/Few_shot/Ai_project/english_data.csv'  # Updated path
df = pd.read_csv(file_path, header=None, names=['raw_data'])

# Split 'raw_data' into 'sentence' and 'type_of_speech' based on the last tab character
df[['sentence', 'type_of_speech']] = df['raw_data'].str.rsplit('\t', n=1, expand=True)
df = df.drop(columns=['raw_data'])

# Clean up whitespace
df['sentence'] = df['sentence'].str.strip()
df['type_of_speech'] = df['type_of_speech'].astype(str).str.strip().replace('nan', '')

# Correct misaligned labels within the sentence
keywords = ["Hope_speech", "Non_hope_speech"]
def extract_label(row):
    for keyword in keywords:
        if keyword in row['sentence']:
            row['type_of_speech'] = keyword
            row['sentence'] = row['sentence'].replace(f"\t{keyword}", "").strip()
    return row
df = df.apply(extract_label, axis=1)

# Remove special characters but keep punctuation
def remove_special_characters(text):
    return re.sub(r'[^A-Za-z0-9\s.,!?\'"-]', '', text)
df['sentence'] = df['sentence'].apply(remove_special_characters)

# Detect and filter out non-English sentences
def is_english(text):
    try:
        return detect(text) == 'en'
    except:
        return False
df = df[df['sentence'].apply(is_english)]

# Save the cleaned dataset
output_path = '/storage/research/data/Few_shot/Ai_project/cleaned_english_data_english_only.csv'  # Updated path
df.to_csv(output_path, index=False)

print(f"Modified dataset saved to {output_path}")



import pandas as pd
import re
from langdetect import detect, DetectorFactory
import os
import matplotlib.pyplot as plt

# Ensure consistent language detection results
DetectorFactory.seed = 0

# Step 1: Load the original dataset
file_path = '/storage/research/data/Few_shot/Ai_project/english_data.csv'  # Updated path
df = pd.read_csv(file_path, header=None, names=['raw_data'])

# Step 2: Split 'raw_data' into 'sentence' and 'type_of_speech' based on the last tab character
df[['sentence', 'type_of_speech']] = df['raw_data'].str.rsplit('\t', n=1, expand=True)
df = df.drop(columns=['raw_data'])

# Step 3: Clean up whitespace and misaligned labels
df['sentence'] = df['sentence'].str.strip()
df['type_of_speech'] = df['type_of_speech'].astype(str).str.strip().replace('nan', '')

# Define keywords indicating 'type_of_speech' labels
keywords = ["Hope_speech", "Non_hope_speech"]

# Function to check and correct misaligned labels within the sentence
def extract_label(row):
    for keyword in keywords:
        if keyword in row['sentence']:
            row['type_of_speech'] = keyword
            row['sentence'] = row['sentence'].replace(f"\t{keyword}", "").strip()
    return row

df = df.apply(extract_label, axis=1)

# Step 4: Remove special characters but keep punctuation
def remove_special_characters(text):
    return re.sub(r'[^A-Za-z0-9\s.,!?\'"-]', '', text)

df['sentence'] = df['sentence'].apply(remove_special_characters)

# Step 5: Filter out non-English sentences
def is_english(text):
    try:
        return detect(text) == 'en'
    except:
        return False

df = df[df['sentence'].apply(is_english)]

# Step 6: Save the cleaned dataset
output_path = '/storage/research/data/Few_shot/Ai_project/cleaned_english_data_english_only.csv'  # Updated path
os.makedirs('/storage/research/data/Few_shot/Ai_project', exist_ok=True)
df.to_csv(output_path, index=False)
print(f"Modified dataset saved to {output_path}")

# Step 7: Basic Exploratory Data Analysis (EDA)

# Dataset dimensions
dataset_shape = df.shape
print("Dataset dimensions:", dataset_shape)

# Count number of instances for each type_of_speech
speech_type_distribution = df['type_of_speech'].value_counts()
print("\nDistribution of type_of_speech:\n", speech_type_distribution)

# Calculate word and character count statistics
df['word_count'] = df['sentence'].str.split().str.len()
df['char_count'] = df['sentence'].str.len()

# Word count statistics
word_count_stats = {
    "Min Word Count": df['word_count'].min(),
    "Max Word Count": df['word_count'].max(),
    "Mean Word Count": df['word_count'].mean(),
    "Median Word Count": df['word_count'].median()
}
print("\nWord Count Statistics:\n", word_count_stats)

# Character count statistics
char_count_stats = {
    "Min Char Count": df['char_count'].min(),
    "Max Char Count": df['char_count'].max(),
    "Mean Char Count": df['char_count'].mean(),
    "Median Char Count": df['char_count'].median()
}
print("\nCharacter Count Statistics:\n", char_count_stats)

# Step 8: Visualizations

# Directory to save the visualizations
visualizations_path = '/storage/research/data/Few_shot/Ai_project/'
os.makedirs(visualizations_path, exist_ok=True)

# Histogram of sentence lengths (in words)
plt.figure(figsize=(10, 6))
plt.hist(df['word_count'], bins=30, edgecolor='k', alpha=0.7)
plt.title("Histogram of Sentence Lengths (Words)")
plt.xlabel("Number of Words")
plt.ylabel("Frequency")
histogram_path = os.path.join(visualizations_path, 'histogram_sentence_lengths.png')
plt.savefig(histogram_path)  # Save the visualization
plt.show()

# Bar chart for distribution of type_of_speech
plt.figure(figsize=(8, 5))
speech_type_distribution.plot(kind='bar', color='skyblue', edgecolor='k')
plt.title("Distribution of Type of Speech")
plt.xlabel("Type of Speech")
plt.ylabel("Count")
plt.xticks(rotation=0)
bar_chart_path = os.path.join(visualizations_path, 'speech_type_distribution.png')
plt.savefig(bar_chart_path)  # Save the visualization
plt.show()

print(f"Visualizations saved to {visualizations_path}")


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from wordcloud import WordCloud
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer
from textstat import flesch_reading_ease
import numpy as np
import os

# Define the base directory for saving outputs
base_path = '/storage/research/data/Few_shot/Ai_project/'
os.makedirs(base_path, exist_ok=True)

# Load the dataset
file_path = os.path.join(base_path, 'cleaned_english_data_english_only.csv')  # Updated path
df = pd.read_csv(file_path)

# Ensure columns for analysis
df['word_count'] = df['sentence'].str.split().str.len()
df['char_count'] = df['sentence'].str.len()

# Analysis and Visualization

# 1. Sentence Length Analysis by Type of Speech
plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x='type_of_speech', y='word_count')
plt.title("Word Count by Type of Speech")
plt.xlabel("Type of Speech")
plt.ylabel("Word Count")
plt.savefig(os.path.join(base_path, 'word_count_by_type_of_speech.png'))
plt.show()

plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x='type_of_speech', y='char_count')
plt.title("Character Count by Type of Speech")
plt.xlabel("Type of Speech")
plt.ylabel("Character Count")
plt.savefig(os.path.join(base_path, 'char_count_by_type_of_speech.png'))
plt.show()

# 2. Most Common Words by Type of Speech
def plot_most_common_words(data, type_of_speech, save_path):
    text = " ".join(data[data['type_of_speech'] == type_of_speech]['sentence'])
    word_counts = Counter(text.split())
    common_words = dict(word_counts.most_common(20))

    plt.figure(figsize=(12, 6))
    plt.bar(common_words.keys(), common_words.values())
    plt.title(f"Most Common Words in {type_of_speech}")
    plt.xticks(rotation=45)
    plt.savefig(save_path)
    plt.show()

plot_most_common_words(df, 'Hope_speech', os.path.join(base_path, 'common_words_hope_speech.png'))
plot_most_common_words(df, 'Non_hope_speech', os.path.join(base_path, 'common_words_non_hope_speech.png'))

# 3. Bigrams and Trigrams Analysis
def plot_top_ngrams(data, ngram_range, title, save_path):
    vectorizer = CountVectorizer(ngram_range=ngram_range, stop_words='english')
    ngrams = vectorizer.fit_transform(data['sentence'])
    ngram_counts = Counter(dict(zip(vectorizer.get_feature_names_out(), ngrams.sum(axis=0).A1)))
    top_ngrams = dict(ngram_counts.most_common(10))

    plt.figure(figsize=(12, 6))
    plt.bar(top_ngrams.keys(), top_ngrams.values())
    plt.title(title)
    plt.xticks(rotation=45)
    plt.savefig(save_path)
    plt.show()

plot_top_ngrams(df[df['type_of_speech'] == 'Hope_speech'], (2, 2), "Top Bigrams in Hope Speech", os.path.join(base_path, 'bigrams_hope_speech.png'))
plot_top_ngrams(df[df['type_of_speech'] == 'Hope_speech'], (3, 3), "Top Trigrams in Hope Speech", os.path.join(base_path, 'trigrams_hope_speech.png'))

plot_top_ngrams(df[df['type_of_speech'] == 'Non_hope_speech'], (2, 2), "Top Bigrams in Non-Hope Speech", os.path.join(base_path, 'bigrams_non_hope_speech.png'))
plot_top_ngrams(df[df['type_of_speech'] == 'Non_hope_speech'], (3, 3), "Top Trigrams in Non-Hope Speech", os.path.join(base_path, 'trigrams_non_hope_speech.png'))

# 4. Sentence Polarity (Sentiment Analysis)
df['sentiment'] = df['sentence'].apply(lambda x: TextBlob(x).sentiment.polarity)
plt.figure(figsize=(8, 6))
sns.barplot(data=df, x='type_of_speech', y='sentiment', ci=None)
plt.title("Average Sentiment by Type of Speech")
plt.xlabel("Type of Speech")
plt.ylabel("Sentiment Score")
plt.savefig(os.path.join(base_path, 'average_sentiment_by_type_of_speech.png'))
plt.show()

# 5. Readability Scores Analysis
df['readability_score'] = df['sentence'].apply(flesch_reading_ease)
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='type_of_speech', y='readability_score')
plt.title("Readability Score by Type of Speech")
plt.xlabel("Type of Speech")
plt.ylabel("Flesch Reading Ease Score")
plt.savefig(os.path.join(base_path, 'readability_score_by_type_of_speech.png'))
plt.show()

# 6. Type of Speech Distribution over Sentence Lengths
plt.figure(figsize=(10, 6))
sns.kdeplot(data=df, x='word_count', hue='type_of_speech', fill=True)
plt.title("Distribution of Sentence Lengths (Words) by Type of Speech")
plt.xlabel("Word Count")
plt.ylabel("Density")
plt.savefig(os.path.join(base_path, 'distribution_sentence_lengths_by_type_of_speech.png'))
plt.show()

# 7. Heatmap of Character and Word Count Correlation
correlation = df[['word_count', 'char_count']].corr()
plt.figure(figsize=(6, 5))
sns.heatmap(correlation, annot=True, cmap="YlGnBu")
plt.title("Correlation between Word and Character Count")
plt.savefig(os.path.join(base_path, 'word_char_count_correlation_heatmap.png'))
plt.show()

# 8. Average Words per Sentence by Type of Speech
avg_word_count = df.groupby('type_of_speech')['word_count'].mean()
plt.figure(figsize=(8, 6))
avg_word_count.plot(kind='bar', color='skyblue', edgecolor='k')
plt.title("Average Words per Sentence by Type of Speech")
plt.xlabel("Type of Speech")
plt.ylabel("Average Word Count")
plt.xticks(rotation=0)
plt.savefig(os.path.join(base_path, 'average_words_per_sentence_by_type_of_speech.png'))
plt.show()

# 9. Outlier Detection
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='word_count', y='char_count', hue='type_of_speech', alpha=0.7)
plt.title("Outliers Detection: Word Count vs. Character Count")
plt.xlabel("Word Count")
plt.ylabel("Character Count")
plt.legend(title="Type of Speech")
plt.savefig(os.path.join(base_path, 'outlier_detection_word_char_count.png'))
plt.show()

print(f"All visualizations saved to {base_path}")

import pandas as pd
import random
import nltk
from nltk.corpus import wordnet
from tqdm import tqdm
import os

# Download necessary NLTK data
nltk.download('wordnet')
nltk.download('omw-1.4')

# Define the base path
base_path = '/storage/research/data/Few_shot/Ai_project/'

# Load the dataset
file_path = os.path.join(base_path, 'cleaned_english_data_english_only.csv')  # Updated path
df = pd.read_csv(file_path)

# Check distribution
print("Original Distribution of type_of_speech:\n", df['type_of_speech'].value_counts())

# Define a function for synonym replacement
def synonym_replacement(sentence, n=1):
    words = sentence.split()
    new_words = words.copy()
    random_word_list = list(set([word for word in words if wordnet.synsets(word)]))
    random.shuffle(random_word_list)

    num_replaced = 0
    for random_word in random_word_list:
        synonyms = wordnet.synsets(random_word)
        if synonyms:
            synonym = synonyms[0].lemmas()[0].name()
            new_words = [synonym if word == random_word else word for word in new_words]
            num_replaced += 1
        if num_replaced >= n:
            break

    return ' '.join(new_words)

# Apply augmentation only on 'Hope_speech' samples
augmented_sentences = []
target_class = 'Hope_speech'
n_augmentations = 5  # Number of augmentations per sentence

# Iterate over each 'Hope_speech' sentence and generate augmented sentences
for sentence in tqdm(df[df['type_of_speech'] == target_class]['sentence']):
    for _ in range(n_augmentations):
        augmented_sentence = synonym_replacement(sentence, n=2)
        augmented_sentences.append([augmented_sentence, target_class])

# Convert augmented data to a DataFrame
augmented_df = pd.DataFrame(augmented_sentences, columns=['sentence', 'type_of_speech'])

# Concatenate the original dataset with the augmented data
balanced_df = pd.concat([df, augmented_df])

# Check new distribution
print("\nNew Distribution of type_of_speech after augmentation:\n", balanced_df['type_of_speech'].value_counts())

# Save the balanced dataset
output_path = os.path.join(base_path, 'balanced_english_data_with_augmentation.csv')  # Updated path
balanced_df.to_csv(output_path, index=False)
print(f"\nBalanced dataset saved to {output_path}")

import pandas as pd
from tqdm import tqdm
import os

# Define the base path
base_path = '/storage/research/data/Few_shot/Ai_project/'

# Load the original dataset
file_path = os.path.join(base_path, 'cleaned_english_data_english_only.csv')  # Updated path
df = pd.read_csv(file_path)

# Define augmentation parameters
target_class = 'Hope_speech'
n_augmentations = 3  # Number of new augmentations per sentence

# Filter the original Hope_speech samples
original_hope_speech_df = df[df['type_of_speech'] == target_class]

# Create additional augmented sentences for only the original Hope_speech samples
additional_augmented_sentences = []

# Generate augmentations only on original "Hope_speech" sentences
for sentence in tqdm(original_hope_speech_df['sentence']):
    for _ in range(n_augmentations):
        augmented_sentence = synonym_replacement(sentence, n=2)  # Assumes synonym_replacement is already defined
        additional_augmented_sentences.append([augmented_sentence, target_class])

# Convert additional augmented data to a DataFrame
additional_augmented_df = pd.DataFrame(additional_augmented_sentences, columns=['sentence', 'type_of_speech'])

# Load the previously balanced dataset and concatenate only the new augmentations
balanced_dataset_path = os.path.join(base_path, 'balanced_english_data_with_augmentation.csv')  # Updated path
balanced_df = pd.read_csv(balanced_dataset_path)
final_balanced_df = pd.concat([balanced_df, additional_augmented_df])

# Check new distribution
print("\nNew Distribution of type_of_speech after second augmentation (limited to original samples):\n", final_balanced_df['type_of_speech'].value_counts())

# Save the final balanced dataset with the second augmentation applied
final_output_path = os.path.join(base_path, 'final_balanced_english_data_with_limited_additional_augmentation.csv')  # Updated path
final_balanced_df.to_csv(final_output_path, index=False)
print(f"\nFinal balanced dataset saved to {final_output_path}")


import pandas as pd
from tqdm import tqdm
import os

# Define the base path
base_path = '/storage/research/data/Few_shot/Ai_project/'

# Load the original dataset
file_path = os.path.join(base_path, 'cleaned_english_data_english_only.csv')  # Updated path
df = pd.read_csv(file_path)

# Define target count for each class
target_count = 20000

# Separate the classes
non_hope_speech_df = df[df['type_of_speech'] == 'Non_hope_speech']
hope_speech_df = df[df['type_of_speech'] == 'Hope_speech']

# Calculate required augmentations for "Hope_speech" to reach the target count
current_hope_speech_count = hope_speech_df.shape[0]
needed_hope_speech_count = target_count - current_hope_speech_count

# If the current "Hope_speech" count is less than the target, augment
if needed_hope_speech_count > 0:
    # Calculate augmentations per sentence to reach the target count
    n_augmentations = needed_hope_speech_count // current_hope_speech_count

    # Generate augmentations for each sentence in the original "Hope_speech"
    additional_augmented_sentences = []
    for sentence in tqdm(hope_speech_df['sentence']):
        for _ in range(n_augmentations):
            augmented_sentence = synonym_replacement(sentence, n=2)  # Assumes synonym_replacement is defined
            additional_augmented_sentences.append([augmented_sentence, 'Hope_speech'])

    # Handle any remaining instances to reach exactly the target count
    remaining_instances = needed_hope_speech_count - (n_augmentations * current_hope_speech_count)
    for sentence in hope_speech_df['sentence'][:remaining_instances]:
        augmented_sentence = synonym_replacement(sentence, n=2)
        additional_augmented_sentences.append([augmented_sentence, 'Hope_speech'])

    # Create DataFrame for augmented data
    additional_augmented_df = pd.DataFrame(additional_augmented_sentences, columns=['sentence', 'type_of_speech'])

    # Combine original "Hope_speech" with augmented data
    final_hope_speech_df = pd.concat([hope_speech_df, additional_augmented_df]).sample(n=target_count, random_state=42)
else:
    # If "Hope_speech" already meets or exceeds the target count, undersample
    final_hope_speech_df = hope_speech_df.sample(n=target_count, random_state=42)

# Undersample "Non_hope_speech" to the target count if necessary
if non_hope_speech_df.shape[0] > target_count:
    final_non_hope_speech_df = non_hope_speech_df.sample(n=target_count, random_state=42)
else:
    final_non_hope_speech_df = non_hope_speech_df

# Combine the balanced "Hope_speech" and "Non_hope_speech" DataFrames
final_balanced_df = pd.concat([final_hope_speech_df, final_non_hope_speech_df])

# Check new distribution
print("\nFinal Distribution of type_of_speech after precise balancing:\n", final_balanced_df['type_of_speech'].value_counts())

# Save the final precisely balanced dataset
final_output_path = os.path.join(base_path, 'final_precisely_balanced_english_data.csv')  # Updated path
final_balanced_df.to_csv(final_output_path, index=False)
print(f"\nPrecisely balanced dataset saved to {final_output_path}")



import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from wordcloud import WordCloud
from textblob import TextBlob
import os

# Define the base path
base_path = '/storage/research/data/Few_shot/Ai_project/'

# Load the dataset
file_path = os.path.join(base_path, 'final_precisely_balanced_english_data.csv')  # Updated path
df = pd.read_csv(file_path)

# Basic structure and summary
print("Dataset Shape:", df.shape)
print("\nDataset Info:")
print(df.info())
print("\nFirst Few Rows:")
print(df.head())

# Ensure necessary columns for analysis
df['word_count'] = df['sentence'].str.split().str.len()
df['char_count'] = df['sentence'].str.len()

# Directory for saving visualizations
visualizations_path = os.path.join(base_path, 'visualizations')
os.makedirs(visualizations_path, exist_ok=True)

# 1. Class Distribution
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='type_of_speech', palette="viridis")
plt.title("Distribution of Type of Speech")
plt.xlabel("Type of Speech")
plt.ylabel("Count")
plt.savefig(os.path.join(visualizations_path, 'class_distribution.png'))
plt.show()

# 2. Sentence Length Analysis (Word Count and Character Count) by Class
plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x='type_of_speech', y='word_count')
plt.title("Word Count by Type of Speech")
plt.xlabel("Type of Speech")
plt.ylabel("Word Count")
plt.savefig(os.path.join(visualizations_path, 'word_count_by_type_of_speech.png'))
plt.show()

plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x='type_of_speech', y='char_count')
plt.title("Character Count by Type of Speech")
plt.xlabel("Type of Speech")
plt.ylabel("Character Count")
plt.savefig(os.path.join(visualizations_path, 'char_count_by_type_of_speech.png'))
plt.show()

# 3. Most Common Words by Class
def plot_most_common_words(data, type_of_speech, n=20, save_path=None):
    text = " ".join(data[data['type_of_speech'] == type_of_speech]['sentence'])
    word_counts = Counter(text.split())
    common_words = dict(word_counts.most_common(n))

    plt.figure(figsize=(12, 6))
    plt.bar(common_words.keys(), common_words.values(), color='skyblue')
    plt.title(f"Most Common Words in {type_of_speech}")
    plt.xticks(rotation=45)
    if save_path:
        plt.savefig(save_path)
    plt.show()

plot_most_common_words(df, 'Hope_speech', save_path=os.path.join(visualizations_path, 'common_words_hope_speech.png'))
plot_most_common_words(df, 'Non_hope_speech', save_path=os.path.join(visualizations_path, 'common_words_non_hope_speech.png'))

# 4. Sentiment Analysis by Class
df['sentiment'] = df['sentence'].apply(lambda x: TextBlob(x).sentiment.polarity)
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='type_of_speech', y='sentiment', palette="coolwarm")
plt.title("Sentiment Analysis by Type of Speech")
plt.xlabel("Type of Speech")
plt.ylabel("Sentiment Score")
plt.savefig(os.path.join(visualizations_path, 'sentiment_analysis_by_type_of_speech.png'))
plt.show()

# 5. Readability Scores by Class (if nltk and textstat installed)
try:
    from textstat import flesch_reading_ease
    df['readability_score'] = df['sentence'].apply(flesch_reading_ease)
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='type_of_speech', y='readability_score', palette="Set2")
    plt.title("Readability Score by Type of Speech")
    plt.xlabel("Type of Speech")
    plt.ylabel("Flesch Reading Ease Score")
    plt.savefig(os.path.join(visualizations_path, 'readability_score_by_type_of_speech.png'))
    plt.show()
except ImportError:
    print("Textstat library is not installed; skipping readability analysis.")

# 6. Word Cloud for Each Class
def plot_wordcloud(data, type_of_speech, save_path=None):
    text = " ".join(data[data['type_of_speech'] == type_of_speech]['sentence'])
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)

    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title(f"Word Cloud for {type_of_speech}")
    if save_path:
        plt.savefig(save_path)
    plt.show()

plot_wordcloud(df, 'Hope_speech', save_path=os.path.join(visualizations_path, 'wordcloud_hope_speech.png'))
plot_wordcloud(df, 'Non_hope_speech', save_path=os.path.join(visualizations_path, 'wordcloud_non_hope_speech.png'))

print(f"All visualizations have been saved to {visualizations_path}")

import nltk
nltk.download('punkt')


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from wordcloud import WordCloud
from textblob import TextBlob
import os

# Define the base path
base_path = '/storage/research/data/Few_shot/Ai_project/'

# Load the dataset
file_path = os.path.join(base_path, 'final_precisely_balanced_english_data.csv')  # Updated path
df = pd.read_csv(file_path)

# Basic structure and summary
print("Dataset Shape:", df.shape)
print("\nDataset Info:")
print(df.info())
print("\nFirst Few Rows:")
print(df.head())

# Ensure necessary columns for analysis
df['word_count'] = df['sentence'].str.split().str.len()
df['char_count'] = df['sentence'].str.len()

# Directory for saving visualizations
visualizations_path = os.path.join(base_path, 'visualizations')
os.makedirs(visualizations_path, exist_ok=True)

# 1. Class Distribution
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='type_of_speech', palette="viridis")
plt.title("Distribution of Type of Speech")
plt.xlabel("Type of Speech")
plt.ylabel("Count")
plt.savefig(os.path.join(visualizations_path, 'class_distribution.png'))
plt.show()

# 2. Sentence Length Analysis (Word Count and Character Count) by Class
plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x='type_of_speech', y='word_count')
plt.title("Word Count by Type of Speech")
plt.xlabel("Type of Speech")
plt.ylabel("Word Count")
plt.savefig(os.path.join(visualizations_path, 'word_count_by_type_of_speech.png'))
plt.show()

plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x='type_of_speech', y='char_count')
plt.title("Character Count by Type of Speech")
plt.xlabel("Type of Speech")
plt.ylabel("Character Count")
plt.savefig(os.path.join(visualizations_path, 'char_count_by_type_of_speech.png'))
plt.show()

# 3. Most Common Words by Class
def plot_most_common_words(data, type_of_speech, n=20, save_path=None):
    text = " ".join(data[data['type_of_speech'] == type_of_speech]['sentence'])
    word_counts = Counter(text.split())
    common_words = dict(word_counts.most_common(n))

    plt.figure(figsize=(12, 6))
    plt.bar(common_words.keys(), common_words.values(), color='skyblue')
    plt.title(f"Most Common Words in {type_of_speech}")
    plt.xticks(rotation=45)
    if save_path:
        plt.savefig(save_path)
    plt.show()

plot_most_common_words(df, 'Hope_speech', save_path=os.path.join(visualizations_path, 'common_words_hope_speech.png'))
plot_most_common_words(df, 'Non_hope_speech', save_path=os.path.join(visualizations_path, 'common_words_non_hope_speech.png'))

# 4. Sentiment Analysis by Class
df['sentiment'] = df['sentence'].apply(lambda x: TextBlob(x).sentiment.polarity)
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='type_of_speech', y='sentiment', palette="coolwarm")
plt.title("Sentiment Analysis by Type of Speech")
plt.xlabel("Type of Speech")
plt.ylabel("Sentiment Score")
plt.savefig(os.path.join(visualizations_path, 'sentiment_analysis_by_type_of_speech.png'))
plt.show()

# 5. Readability Scores by Class (if nltk and textstat installed)
try:
    from textstat import flesch_reading_ease
    df['readability_score'] = df['sentence'].apply(flesch_reading_ease)
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='type_of_speech', y='readability_score', palette="Set2")
    plt.title("Readability Score by Type of Speech")
    plt.xlabel("Type of Speech")
    plt.ylabel("Flesch Reading Ease Score")
    plt.savefig(os.path.join(visualizations_path, 'readability_score_by_type_of_speech.png'))
    plt.show()
except ImportError:
    print("Textstat library is not installed; skipping readability analysis.")

# 6. Word Cloud for Each Class
def plot_wordcloud(data, type_of_speech, save_path=None):
    text = " ".join(data[data['type_of_speech'] == type_of_speech]['sentence'])
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)

    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title(f"Word Cloud for {type_of_speech}")
    if save_path:
        plt.savefig(save_path)
    plt.show()

plot_wordcloud(df, 'Hope_speech', save_path=os.path.join(visualizations_path, 'wordcloud_hope_speech.png'))
plot_wordcloud(df, 'Non_hope_speech', save_path=os.path.join(visualizations_path, 'wordcloud_non_hope_speech.png'))

print(f"All visualizations have been saved to {visualizations_path}")

import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from wordcloud import WordCloud
import numpy as np
import os

# Define the base path for saving visualizations
base_path = '/storage/research/data/Few_shot/Ai_project/'
visualizations_path = os.path.join(base_path, 'visualizations')
os.makedirs(visualizations_path, exist_ok=True)

# 1. Correlation Matrix for Numerical Features
numerical_df = df[['word_count', 'char_count', 'sentiment']]
if 'readability_score' in df.columns:
    numerical_df['readability_score'] = df['readability_score']

plt.figure(figsize=(8, 6))
sns.heatmap(numerical_df.corr(), annot=True, cmap="YlGnBu")
plt.title("Correlation Matrix of Numerical Features")
plt.savefig(os.path.join(visualizations_path, 'correlation_matrix.png'))
plt.show()

# 2. Average Word Length by Class
df['avg_word_length'] = df['sentence'].apply(lambda x: np.mean([len(word) for word in x.split()]))
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='type_of_speech', y='avg_word_length', palette="pastel")
plt.title("Average Word Length by Type of Speech")
plt.xlabel("Type of Speech")
plt.ylabel("Average Word Length")
plt.savefig(os.path.join(visualizations_path, 'average_word_length_by_type.png'))
plt.show()

# 3. Most Common Words by Sentiment (positive vs. negative)
positive_text = " ".join(df[df['sentiment'] > 0]['sentence'])
negative_text = " ".join(df[df['sentiment'] <= 0]['sentence'])

# Word Cloud for Positive Sentences
positive_wordcloud = WordCloud(width=800, height=400, background_color="white").generate(positive_text)
plt.figure(figsize=(10, 6))
plt.imshow(positive_wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title("Most Common Words in Positive Sentences")
plt.savefig(os.path.join(visualizations_path, 'positive_wordcloud.png'))
plt.show()

# Word Cloud for Negative Sentences
negative_wordcloud = WordCloud(width=800, height=400, background_color="white").generate(negative_text)
plt.figure(figsize=(10, 6))
plt.imshow(negative_wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title("Most Common Words in Negative Sentences")
plt.savefig(os.path.join(visualizations_path, 'negative_wordcloud.png'))
plt.show()

# 4. Sentiment vs. Readability Score (if readability is available)
if 'readability_score' in df.columns:
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='readability_score', y='sentiment', hue='type_of_speech', alpha=0.7)
    plt.title("Sentiment vs. Readability Score by Type of Speech")
    plt.xlabel("Readability Score (Flesch Reading Ease)")
    plt.ylabel("Sentiment Score")
    plt.legend(title="Type of Speech")
    plt.savefig(os.path.join(visualizations_path, 'sentiment_vs_readability.png'))
    plt.show()

# 5. Top Words by Average Sentiment
all_words = df['sentence'].str.split().explode()
word_sentiments = df[['sentence', 'sentiment']].copy()
word_sentiments['words'] = word_sentiments['sentence'].str.split()
word_sentiments = word_sentiments.explode('words')
average_sentiment_per_word = word_sentiments.groupby('words')['sentiment'].mean().sort_values()

# Plot top positive and negative words
top_positive_words = average_sentiment_per_word[-10:]
top_negative_words = average_sentiment_per_word[:10]

plt.figure(figsize=(12, 6))
plt.bar(top_positive_words.index, top_positive_words.values, color='green')
plt.title("Top Words by Positive Sentiment")
plt.xlabel("Words")
plt.ylabel("Average Sentiment")
plt.xticks(rotation=45)
plt.savefig(os.path.join(visualizations_path, 'top_positive_words.png'))
plt.show()

plt.figure(figsize=(12, 6))
plt.bar(top_negative_words.index, top_negative_words.values, color='red')
plt.title("Top Words by Negative Sentiment")
plt.xlabel("Words")
plt.ylabel("Average Sentiment")
plt.xticks(rotation=45)
plt.savefig(os.path.join(visualizations_path, 'top_negative_words.png'))
plt.show()

print(f"All visualizations have been saved to {visualizations_path}")

