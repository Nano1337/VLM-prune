import pandas as pd


tenk_filtered_parquet = "tenk_filtered.parquet"
clip_scores_parquet = "clip_scores.parquet"
vqa_scores_parquet = "vqa_scores_valid_sorted.parquet"

# Load the dataframes from the parquet files
tenk_filtered_df = pd.read_parquet(tenk_filtered_parquet)
clip_scores_df = pd.read_parquet(clip_scores_parquet)
vqa_scores_df = pd.read_parquet(vqa_scores_parquet)

# Merge the dataframes on the 'uid' column
merged_df = tenk_filtered_df.merge(clip_scores_df, on='uid', how='inner')
merged_df = merged_df.merge(vqa_scores_df, on='uid', how='inner')

# # Save the merged dataframe to a new parquet file
# merged_df.to_parquet('merged_10k_results.parquet', index=False)

import matplotlib.pyplot as plt

# Plot histogram of CLIPscore column with range 0.0 to 1.0 at bin increments of 0.02
plt.hist(merged_df['CLIPscore'], bins=[i * 0.02 for i in range(51)], range=(0.0, 1.0), edgecolor='black')
plt.xlabel('CLIPscore')
plt.ylabel('Frequency')
plt.title('Histogram of CLIPscore for 10k samples')
plt.grid(True)

# Save the histogram as a PNG file
plt.savefig('clip_scores_histogram_10k.png')
plt.close()

# Plot histogram for the top 30% CLIPscores
top_30_percent_threshold = merged_df['CLIPscore'].quantile(0.7)
df_top_30_percent = merged_df[merged_df['CLIPscore'] >= top_30_percent_threshold]
print(f"Top 30% threshold: {top_30_percent_threshold}")
print(f"Number of rows in top 30%: {len(df_top_30_percent)}")

plt.hist(df_top_30_percent['CLIPscore'], bins=[i * 0.02 for i in range(51)], range=(0.0, 1.0), edgecolor='black')
plt.xlabel('CLIPscore')
plt.ylabel('Frequency')
plt.title('Histogram of Top 30% CLIPscore for 10k samples')
plt.grid(True)

# Save the histogram as a PNG file
plt.savefig('clip_scores_histogram_top_30_percent_10k.png')
plt.close()

# Find the uids corresponding to the top 30 percent of CLIPscore
clip_top_30_percent_threshold = merged_df['CLIPscore'].quantile(0.7)
clip_top_30_percent_uids = set(merged_df[merged_df['CLIPscore'] >= clip_top_30_percent_threshold]['uid'])

# Find the uids corresponding to the top 30 percent of VQAscore
vqa_top_30_percent_threshold = merged_df['VQAscore'].quantile(0.7)
vqa_top_30_percent_uids = set(merged_df[merged_df['VQAscore'] >= vqa_top_30_percent_threshold]['uid'])

# Calculate the fractional intersection of the uids
intersection_uids = clip_top_30_percent_uids.intersection(vqa_top_30_percent_uids)
fractional_intersection = len(intersection_uids) / len(clip_top_30_percent_uids.union(vqa_top_30_percent_uids))

print(f"Fractional intersection of top 30% CLIPscore and VQAscore uids: {fractional_intersection}")
