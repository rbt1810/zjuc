import fitz  # PyMuPDF
import numpy as np
from PIL import Image
import matplotlib.pyplot as plot
import sys

def extract_pdf_formulas(pdf_path):
	doc = fitz.open(pdf_path)
	formula_images = []

	for page_num in range(len(doc)):
		page = doc.load_page(page_num)
        # 方法1：提取原生LaTeX
		text_instances = page.get_text("dict")["blocks"]
		for block in text_instances:
			if "formula" in block:  # 检测公式特征
				formula = block["text"]
				formula_images.append(render_latex(formula))  # 转换为图片
	return formula_images

def render_latex(formula):
    # 使用matplotlib生成公式图片
    plt.axis('off')
    plt.text(0.5, 0.5, f"${formula}$", size=20, ha='center', va='center')
    plt.gcf().canvas.draw()
    img = np.frombuffer(plt.gcf().canvas.tobytes(), dtype=np.uint8)
    return img.reshape(plt.gcf().canvas.get_width_height()[::-1] + (4,))[..., :3]

if __name__ == '__main__':
	images = extract_pdf_formulas(f'./lectures/{sys.argv[1]}/{sys.argv[1]}.pdf')
	print(images)
