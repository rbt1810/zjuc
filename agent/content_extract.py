from pypdf import PdfReader

reader = PdfReader("Lecture_ppt.pdf")
text = ""
for page in reader.pages:
    text += page.extract_text() or ""

with open('Text.txt', 'w') as f:
    f.write(text)
print('Done')
