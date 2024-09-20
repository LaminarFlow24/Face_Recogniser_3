import os
import pandas as pd

# Specify the folder path
folder_path = './'

# Create an empty DataFrame with the specified columns in YYYY/MM/DD format
columns = ['students'] + [f'2024-09-{str(day).zfill(2)}' for day in range(1, 31)]  # Adjust the year if needed 
df = pd.DataFrame(columns=columns)

# Iterate over all subfolders in the specified folder
subfolders = [name for name in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, name))]

# Add the subfolder names to the 'students' column
df['students'] = subfolders

# Save the DataFrame to an Excel file
output_file = 'students_attendance_september.xlsx'
df.to_excel(output_file, index=False)

print(f"Excel file created: {output_file}")
