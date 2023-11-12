import pandas as pd
import argparse
import json

def remove_columns(input_csv, output_csv, columns_to_remove):
    # Read input CSV file
    data = pd.read_csv(input_csv)

    # Remove specified columns
    data.drop(columns=columns_to_remove, inplace=True, errors='ignore')

    # Save the modified data to output CSV file
    data.to_csv(output_csv, index=False)
    print(f"Columns removed and data saved to '{output_csv}'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Remove specified columns from a CSV file.')
    parser.add_argument('--input_csv', type=str, help='Path to input CSV file')
    parser.add_argument('--output_csv', type=str, help='Path to save the modified CSV file')
    parser.add_argument('--columns', nargs='+', help='Columns to remove')
    args = parser.parse_args()

    # Load column names to remove from JSON file
    # with open(args.columns_json, 'r') as json_file:
    #     columns_to_remove = json.load(json_file)

    remove_columns(args.input_csv, args.output_csv, args.columns)
