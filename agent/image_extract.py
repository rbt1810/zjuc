from pypdf import PdfReader
import sys

reader = PdfReader(f"lectures/{sys.argv[1]}/{sys.argv[1]}.pdf")
page = reader.pages
for idx in page:
    for img in idx.images:
        with open(f"lectures/{sys.argv[1]}/Images/image_{img.name}", "wb") as f:
            f.write(img.data)

