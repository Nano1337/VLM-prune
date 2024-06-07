import pandas as pd


tenk_filtered_parquet = "tenk_filtered_english.parquet"
clip_scores_parquet = "clip_scores.parquet"
vqa_scores_parquet = "vqa_scores_valid_sorted.parquet"

# Load the dataframes from the parquet files
tenk_filtered_df = pd.read_parquet(tenk_filtered_parquet)
clip_scores_df = pd.read_parquet(clip_scores_parquet)
vqa_scores_df = pd.read_parquet(vqa_scores_parquet)

# Take the intersection of 'uid' across all dataframes
common_uids = set(tenk_filtered_df['uid']).intersection(clip_scores_df['uid']).intersection(vqa_scores_df['uid'])

# Filter dataframes to only include common uids
tenk_filtered_df = tenk_filtered_df[tenk_filtered_df['uid'].isin(common_uids)]
clip_scores_df = clip_scores_df[clip_scores_df['uid'].isin(common_uids)]
vqa_scores_df = vqa_scores_df[vqa_scores_df['uid'].isin(common_uids)]

# Merge the dataframes on the 'uid' column
merged_df = tenk_filtered_df.merge(clip_scores_df, on='uid')
merged_df = merged_df.merge(vqa_scores_df, on='uid')

# # Save the merged dataframe to a new parquet file
# merged_df.to_parquet('merged_3k_results.parquet', index=False)

# import matplotlib.pyplot as plt

# # Plot histogram of CLIPscore column with range 0.0 to 1.0 at bin increments of 0.02
# plt.hist(merged_df['VQAscore'], bins=[i * 0.02 for i in range(51)], range=(0.0, 1.0), edgecolor='black')
# plt.xlabel('VQAscore')
# plt.ylabel('Frequency')
# plt.title('Histogram of VQAscore for 3k English samples')
# plt.grid(True)

# # Save the histogram as a PNG file
# plt.savefig('vqa_scores_histogram_3k.png')
# plt.close()

# exit()

# # Plot histogram for the top 30% CLIPscores
# top_30_percent_threshold = merged_df['CLIPscore'].quantile(0.7)
# df_top_30_percent = merged_df[merged_df['CLIPscore'] >= top_30_percent_threshold]
# print(f"Top 30% threshold: {top_30_percent_threshold}")
# print(f"Number of rows in top 30%: {len(df_top_30_percent)}")

# plt.hist(df_top_30_percent['CLIPscore'], bins=[i * 0.02 for i in range(51)], range=(0.0, 1.0), edgecolor='black')
# plt.xlabel('CLIPscore')
# plt.ylabel('Frequency')
# plt.title('Histogram of Top 30% CLIPscore for 10k samples')
# plt.grid(True)

# # Save the histogram as a PNG file
# plt.savefig('clip_scores_histogram_top_30_percent_10k.png')
# plt.close()


### Calculating the top 30 percentile

# # Find the uids corresponding to the top 30 percent of CLIPscore
# clip_top_30_percent_threshold = merged_df['CLIPscore'].quantile(0.7)
# clip_top_30_percent_uids = set(merged_df[merged_df['CLIPscore'] >= clip_top_30_percent_threshold]['uid'])

# # Find the uids corresponding to the top 30 percent of VQAscore
# vqa_top_30_percent_threshold = merged_df['VQAscore'].quantile(0.7)
# vqa_top_30_percent_uids = set(merged_df[merged_df['VQAscore'] >= vqa_top_30_percent_threshold]['uid'])

# # Calculate the fractional intersection of the uids
# intersection_uids = clip_top_30_percent_uids.intersection(vqa_top_30_percent_uids)
# fractional_intersection = len(intersection_uids) / len(clip_top_30_percent_uids.union(vqa_top_30_percent_uids))

# print(f"Fractional intersection of top 30% CLIPscore and VQAscore uids: {fractional_intersection}")
# Calculate percentiles for CLIPscore and VQAscore
merged_df['CLIPscore_percentile'] = merged_df['CLIPscore'].rank(pct=True)
merged_df['VQAscore_percentile'] = merged_df['VQAscore'].rank(pct=True)

# Calculate the absolute value of the relative percentile difference
merged_df['percentile_difference'] = (merged_df['CLIPscore_percentile'] - merged_df['VQAscore_percentile']).abs()

# Sort the DataFrame by the 'percentile_difference' column
sorted_df = merged_df.sort_values(by='percentile_difference')

# Print the first few rows of the sorted DataFrame to verify
print(sorted_df[['uid', 'percentile_difference', 'CLIPscore', 'VQAscore', 'CLIPscore_percentile', 'VQAscore_percentile', 'caption']].head(10))

# Sort the DataFrame by the 'percentile_difference' column in reverse order
sorted_df_reverse = merged_df.sort_values(by='percentile_difference', ascending=False)
import unicodedata

# Function to truncate caption based on display width considering multilingual characters
def truncate_caption(caption, max_width=80):
    current_width = 0
    truncated_caption = ""
    for char in caption:
        char_width = unicodedata.east_asian_width(char)
        if char_width in "FWA": # Fullwidth or Wide or Ambiguous could be wider
            current_width += 2
        else:
            current_width += 1
        if current_width > max_width:
            truncated_caption = truncated_caption.rstrip() + '...'
            break
        truncated_caption += char
    return truncated_caption

# Save the first 16 rows of sorted_df to a CSV file
sorted_df.head(16).to_csv('top_16_agree_rows.csv', index=False)

# Adjust captions to ensure they fit within the table layout without overflow, considering multilingual characters
sorted_df['caption'] = sorted_df['caption'].apply(truncate_caption)

# Display the top 16 rows of the DataFrame with adjusted captions for verification
print(sorted_df[['uid', 'percentile_difference', 'CLIPscore', 'VQAscore', 'CLIPscore_percentile', 'VQAscore_percentile', 'caption']].head(16).to_string(index=False))
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Define the image directory
image_directory = '/home/haoli/VLM-prune/data/sampled_datacomp/sampled_datacomp_images'

# Get the uids and keys of the first 16 rows
top_16_uids_keys = sorted_df[['uid', 'key', 'CLIPscore_percentile', 'VQAscore_percentile']].head(16)
# Create a figure with a 4x4 grid
fig, axs = plt.subplots(4, 4, figsize=(20, 16))

# Flatten the array of axes for easy iteration
axs = axs.ravel()

# Loop through the top 16 uids and keys, loading and plotting each image
for idx, (uid, key, clipscore_percentile, vqascore_percentile) in enumerate(top_16_uids_keys.itertuples(index=False)):
    image_path = f"{image_directory}/{key}.jpg"
    try:
        img = mpimg.imread(image_path)
        axs[idx].imshow(img)
        # Determine title color based on which percentile is higher
        if vqascore_percentile > clipscore_percentile:
            title_color = 'black' # green
        else:
            title_color = 'black' # blue
        axs[idx].set_title(f"UID: {uid}", color=title_color)
        axs[idx].axis('off')  # Hide axes
    except FileNotFoundError:
        axs[idx].set_title(f"Image not found for UID: {uid}", color='red')
        axs[idx].axis('off')  # Hide axes

# Adjust layout to prevent overlap
plt.tight_layout()
# Save the plot to a file
plt.savefig('top_16_agree_images_plot.png')
