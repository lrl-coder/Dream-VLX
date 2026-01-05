import os

def calculate_average_acc(directory):
    acc_values = []
    all_ids = set(list(range(50)))
    cur_ids = set()
    # Iterate over all files in the directory
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        
        # Skip directories, only process files
        if not os.path.isfile(filepath):
            continue
            
        last_acc_line = None
        
        # Read the file and find the last occurrence of the target line
        is_done = False
        with open(filepath, 'r') as file:
            for line in file:
                if "Overall success rate: " in line:
                    last_acc_line = line.strip()
                if "Done!" in line:
                    is_done = True
        # if not is_done:
        #     raise RuntimeError("Evaluation not done")
        # If we found the line, extract the ACC value
        if last_acc_line:
            try:
                # Split the line and get the last part (the number)
                acc_str = last_acc_line.split("Overall success rate: ")[1].split(' ')[0]
                acc = float(acc_str)
                acc_values.append(acc)
                cur_ids.add(int(filename.split('trial_')[1].split('.')[0]))
            except (IndexError, ValueError) as e:
                print(f"Error processing file {filename}: {e}")
                continue
    
    # Calculate the average
    if acc_values:
        average_acc = sum(acc_values) / len(acc_values)
        print(acc_values)
        print(f"Processed {len(acc_values)} files. Average ACC: {average_acc:.4f}")
        if len(cur_ids) != len(all_ids):
            print("missing: ", all_ids.difference(cur_ids))
        return average_acc
    else:
        print("No ACC values found in any files.")
        return 0.0

import sys
# Example usage:
directory_path = sys.argv[1]
calculate_average_acc(directory_path)