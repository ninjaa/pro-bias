import csv
import os


def filter_mapped_rater_scores():
    # Ensure the scripts directory exists
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(os.path.dirname(script_dir), 'datasets', 'aes')

    # Read the essay_ids from representative_samples.csv
    representative_ids = set()
    with open(os.path.join(data_dir, 'representative_samples.csv'), 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            representative_ids.add(row['essay_id'])

    # Read and write the filtered data
    input_path = os.path.join(data_dir, 'mapped_rater_scores.csv')
    output_path = os.path.join(data_dir, 'mapped_rater_scores_filtered.csv')

    with open(input_path, 'r') as input_file, \
            open(output_path, 'w', newline='') as output_file:
        reader = csv.DictReader(input_file)
        writer = csv.DictWriter(output_file, fieldnames=reader.fieldnames)

        writer.writeheader()
        for row in reader:
            if row['essay_id'] not in representative_ids:
                writer.writerow(row)

    print(f"Filtered file created: {output_path}")


if __name__ == "__main__":
    filter_mapped_rater_scores()
