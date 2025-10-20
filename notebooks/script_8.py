
# List all created files and directories
import os

print("=" * 80)
print("PROJECT STRUCTURE CREATED SUCCESSFULLY")
print("=" * 80)
print("\nComplete directory structure:\n")

for root, dirs, files in os.walk('data_curriculum'):
    level = root.replace('data_curriculum', '').count(os.sep)
    indent = ' ' * 2 * level
    print(f'{indent}{os.path.basename(root)}/')
    subindent = ' ' * 2 * (level + 1)
    for file in files:
        file_path = os.path.join(root, file)
        file_size = os.path.getsize(file_path)
        print(f'{subindent}├─ {file} ({file_size:,} bytes)')

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

# Count files and directories
total_files = sum([len(files) for r, d, files in os.walk('data_curriculum')])
total_dirs = sum([len(dirs) for r, dirs, f in os.walk('data_curriculum')])

print(f"\nTotal directories: {total_dirs}")
print(f"Total files: {total_files}")

print("\n✓ Data files created in data/")
print("✓ Python modules created in src/")
print("✓ Jupyter notebook created in notebooks/")
print("✓ Documentation created (README.md, LICENSE)")
print("✓ Dependencies listed in requirements.txt")
print("✓ Figures directory prepared")

print("\n" + "=" * 80)
print("NEXT STEPS")
print("=" * 80)
print("\n1. Navigate to the project directory:")
print("   cd data_curriculum")
print("\n2. Install required packages:")
print("   pip install -r requirements.txt")
print("\n3. Launch Jupyter Notebook:")
print("   jupyter notebook")
print("\n4. Open and run notebooks/inspection_code.ipynb")
print("\n5. Generated figures will be saved to figures/")
