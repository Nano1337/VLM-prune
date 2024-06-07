
import pandas as pd

# Load the parquet file into a DataFrame
df = pd.read_parquet('vqa_scores.parquet')

# Delete rows where the VQAscore is -1
df = df[df['VQAscore'] != -1]

# Print the total number of rows left after deletion
print(f"Total number of rows left after deletion: {len(df)}")

# instead of deleting, we can just save the valid rows to a new parquet file
df_valid_sorted = df[df['VQAscore'] != -1].sort_values(by='VQAscore', ascending=False)
print(df_valid_sorted.head(10))

import matplotlib.pyplot as plt

# Plot histogram of VQAscore column with range 0.0 to 1.0 at bin increments of 0.02
plt.hist(df_valid_sorted['VQAscore'], bins=[i * 0.02 for i in range(51)], range=(0.0, 1.0), edgecolor='black')
plt.xlabel('VQAscore')
plt.ylabel('Frequency')
plt.title('Histogram of VQAscore for 10k samples')
plt.grid(True)

# Save the histogram as a PNG file
plt.savefig('vqa_scores_histogram_10k.png')
plt.close()

# Plot histogram for the top 30% VQAscores
top_30_percent_threshold = df_valid_sorted['VQAscore'].quantile(0.7)
df_top_30_percent = df_valid_sorted[df_valid_sorted['VQAscore'] >= top_30_percent_threshold]
print(f"Top 30% threshold: {top_30_percent_threshold}")
print(f"Number of rows in top 30%: {len(df_top_30_percent)}")

plt.hist(df_top_30_percent['VQAscore'], bins=[i * 0.02 for i in range(51)], range=(0.0, 1.0), edgecolor='black')
plt.xlabel('VQAscore')
plt.ylabel('Frequency')
plt.title('Histogram of Top 30% VQAscore for 10k samples')
plt.grid(True)

# Save the histogram as a PNG file
plt.savefig('vqa_scores_histogram_top_30_percent_10k.png')
plt.close()

# save filtered
df_valid_sorted.to_parquet('vqa_scores_valid_sorted.parquet', index=False) 

og_parquet_file_path = "data/sampled_datacomp/tenk_dataset.parquet"
og_unfiltered_df = pd.read_parquet(og_parquet_file_path)

# Find the intersection of uids between og_unfiltered_df and df_valid_sorted
intersection_uids = set(og_unfiltered_df['uid']).intersection(set(df_valid_sorted['uid']))

# Filter both DataFrames to only keep rows with uids in the intersection
og_filtered_df = og_unfiltered_df[og_unfiltered_df['uid'].isin(intersection_uids)]
df_valid_sorted = df_valid_sorted[df_valid_sorted['uid'].isin(intersection_uids)]

# Print the number of rows in the intersection
print(f"Number of rows in the intersection (sanity check): {len(intersection_uids)}")
assert len(intersection_uids) == len(og_filtered_df) == len(df_valid_sorted)

# save filtered data to parquet
og_filtered_df.to_parquet('tenk_filtered.parquet', index=False)

# # show the first 10 rows of the filtered data
# print(og_filtered_df.head(10))
# print(df_valid_sorted.head(10))

