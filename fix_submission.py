import pandas as pd
import os
import glob

def fix_submission_file():
    """
    Merges the sample_id from test.csv with the price from the most recent
    submission file and saves it to a new corrected file.
    """
    submission_dir = 'submission'
    test_file_path = 'dataset/test.csv'

    # 1. Find the most recent submission file
    list_of_files = glob.glob(os.path.join(submission_dir, 'submission_*.csv'))
    if not list_of_files:
        print(f"Error: No submission file found in the '{submission_dir}' directory.")
        return

    latest_file = max(list_of_files, key=os.path.getctime)
    print(f"Found latest submission file: {latest_file}")

    output_file_path = os.path.join(submission_dir, 'submission_fixed.csv')

    try:
        # 2. Read the necessary columns
        print("Reading source files...")
        test_df = pd.read_csv(test_file_path, usecols=['sample_id'])
        submission_df = pd.read_csv(latest_file, usecols=['price'])

        # 3. Check for consistency and combine
        if len(test_df) == len(submission_df):
            print("Row counts match. Merging files...")
            final_df = pd.concat([test_df, submission_df], axis=1)
            
            # 4. Save the new file
            final_df.to_csv(output_file_path, index=False)
            print(f"Successfully created fixed submission file at: {output_file_path}")
        else:
            print("Error: Row count mismatch between test.csv and the submission file.")
            print(f"Rows in {test_file_path}: {len(test_df)}")
            print(f"Rows in {latest_file}: {len(submission_df)}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    fix_submission_file()
