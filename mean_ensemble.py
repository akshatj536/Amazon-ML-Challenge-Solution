import pandas as pd
import os
import pandas as pd 
import numpy as np


dberta = pd.read_csv(os.path.join( 'dberta_submission.csv'))
neoberta = pd.read_csv(os.path.join( 'neobert_final_submission.csv'))
rexbert = pd.read_csv(os.path.join( 'rexbert_submission.csv'))



# Compute the average across all three
avg_df = (dberta + neoberta + rexbert) / 3

# Optional: round values for neatness
avg_df = avg_df.round(2)

# Save to a new CSV file
avg_df.to_csv('average_output.csv', index=False)

print("✅ Averaged CSV saved as 'average_output.csv'")
