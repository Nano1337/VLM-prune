import fasttext
from huggingface_hub import hf_hub_download
import re
import pandas as pd
file_path = "tenk_filtered.parquet"
original_df = pd.read_parquet(file_path)
df = pd.read_parquet(file_path)

model_path = hf_hub_download(repo_id="facebook/fasttext-language-identification", filename="model.bin")
model = fasttext.load_model(model_path)

# Predict the language of each caption in the DataFrame using batched predictions
def predict_batch(captions, batch_size=100):
    results = []
    for i in range(0, len(captions), batch_size):
        batch = captions[i:i+batch_size]
        # Remove newline characters from each caption in the batch
        batch = [caption.replace('\n', ' ') for caption in batch]
        predictions, _ = model.predict(batch)
        results.extend([pred[0] for pred in predictions])
    return results

df['lang_predictions'] = predict_batch(df['caption'].tolist())
df = df[df['lang_predictions'].apply(lambda x: x == '__label__eng_Latn')]

# Define a regex pattern for Chinese characters
chinese_char_pattern = re.compile(r'[\u4e00-\u9fff]+')

# Filter out any captions that contain Chinese characters
df = df[~df['caption'].str.contains(chinese_char_pattern)]

# get rid of lang predictions column
df = df.drop(columns=['lang_predictions'])

# save df in new parquet file
df.to_parquet('tenk_filtered_english.parquet')
