import os, re

def count_lines_in_files(directory, regex_pattern):
    total_lines = 0

    # Walk through the directory recursively
    for root, dirs, files in os.walk(directory):
        for file_name in files:
            file_path = os.path.join(root, file_name)

            # Check if the file name matches the regular expression
            if re.search(regex_pattern, file_name):
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                    # Count the number of lines in the file
                    lines = sum(1 for line in file)
                    total_lines += lines
                    print(f"File: {file_path}, Lines: {lines}")

    print(f"\nTotal lines in files matching the pattern: {total_lines}")

# Example usage:
directory_path = "./"
regex_pattern = r'\.cu$'  # Example: Count lines in files with a .txt extension

count_lines_in_files(directory_path, regex_pattern)