from unstructured.partition.auto import partition
import sys
elements = partition(f"lectures/{sys.argv[1]}/{sys.argv[1]}.pdf", strategy="auto")
with open(f'lectures/{sys.argv[1]}/fined_text.txt', 'a') as f:
    for el in elements:
        f.write(str(el))
        f.write('\n\n')
