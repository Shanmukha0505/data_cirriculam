
import os

# Create the directory structure
directories = [
    'data_curriculum/data',
    'data_curriculum/notebooks',
    'data_curriculum/src',
    'data_curriculum/figures'
]

for directory in directories:
    os.makedirs(directory, exist_ok=True)
    print(f"Created: {directory}")

print("\nDirectory structure created successfully!")
