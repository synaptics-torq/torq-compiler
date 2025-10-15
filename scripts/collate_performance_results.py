#!/usr/bin/env python3
import os
import pandas as pd
import argparse


def load_csv_files_from_dir(directory):
    csv_files = [f for f in os.listdir(directory) if f.endswith('.csv')]
    dataframes = {}
    for csv_file in csv_files:
        path = os.path.join(directory, csv_file)
        dataframes[csv_file] = pd.read_csv(path, sep=';')

        if 'location' in dataframes[csv_file].columns:
            dataframes[csv_file] = dataframes[csv_file].drop(columns=['location'])

    return dataframes

def summary(dataframes):
    summary_list = []
    for name, df in dataframes.items():
        total_time = df['time_since_start'].sum()
        summary_list.append({'name': name, 'total_time_since_start': total_time})

    return pd.DataFrame(summary_list)


def main():
    parser = argparse.ArgumentParser(description="Collate CSV files from a directory into an Excel file.")
    parser.add_argument("directory", help="Directory containing CSV files")
    parser.add_argument("output_excel_file", help="Path to output Excel file")
    args = parser.parse_args()

    dataframes = load_csv_files_from_dir(args.directory)

    if not args.output_excel_file.lower().endswith(('.xlsx', '.xls')):
        raise ValueError("Output file must have a .xlsx or .xls extension")

    with pd.ExcelWriter(args.output_excel_file) as writer:

        summary_df = summary(dataframes)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)

        for name, df in dataframes.items():
            sheet_name = os.path.splitext(name)[0][-31:]  # Excel sheet names max 31 chars
            df.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"Excel file saved to {args.output_excel_file}")
    
if __name__ == "__main__":
    main()    
