import csv
import argparse

def parse_csv(filename):
    """Parse CSV file with operation_name, input_size, output_size, and location columns."""
    data = []
    
    with open(filename, 'r') as file:
        reader = csv.DictReader(file, fieldnames=['operation_name', 'input_size', 'output_size', 'location'])
        for row in reader:
            data.append({
                'operation_name': row['operation_name'],
                'input_size': row['input_size'],
                'output_size': row['output_size'],
                'location': row['location']
            })
    
    return data


def main():
    parser = argparse.ArgumentParser(description='Parse tensor sizes CSV file')
    parser.add_argument('csv_file', help='Path to CSV file')
    parser.add_argument('--lram-size', type=int, default=512000, help='LRAM size in bytes')
    args = parser.parse_args()
    
    data = parse_csv(args.csv_file)

    total_large_layers = 0
    total_large_input_layers = 0

    # Print parsed data
    for entry in data:

        if entry['input_size'] == 'unknown' or entry['output_size'] == 'unknown':
            continue 

        input_size = int(entry['input_size'])
        output_size = int(entry['output_size'])

        if input_size + output_size < args.lram_size:
            continue

        total_large_layers += 1

        if input_size > args.lram_size:
            total_large_input_layers += 1

    print(f"Total layers with input/outputs size larger than {args.lram_size} bytes: {total_large_layers}")
    print(f"Total layers with input size larger than {args.lram_size} bytes: {total_large_input_layers}")
        


if __name__ == '__main__':
    main()