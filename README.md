# Shipping Label OCR Text Extraction (AI/ML Assessment Task)

This project is an OCR-based text extraction system for shipping labels / waybills.

## 🎯 Objective

- Perform OCR on shipping label images.
- Extract the **complete text line containing the pattern `_1_`**.
- Achieve high accuracy (target ≥ 75%) on the test set.
- Provide a simple Streamlit UI for demo.

---

## 🧱 Tech Stack

- **Language**: Python
- **OCR Engine**: [EasyOCR](https://github.com/JaidedAI/EasyOCR) (open-source)
- **Image Processing**: OpenCV, NumPy
- **Frontend**: Streamlit
- **Testing**: pytest
- **Logging**: loguru

> No commercial APIs (Google Vision, AWS Textract, Azure OCR, etc.) are used.

---

## 📁 Project Structure

```text
ai-ml-ocr-assessment/
├── README.md
├── requirements.txt
├── app.py                 # Streamlit app
├── src/
│   ├── __init__.py        # Makes src a package
│   ├── ocr_engine.py      # OCR engine wrapper
│   ├── preprocessing.py   # Image pre-processing
│   ├── text_extraction.py # Target line extraction & accuracy
│   └── utils.py           # Logging, helpers
├── tests/
│   ├── __init__.py
│   ├── test_text_extraction.py
│   └── test_utils.py
├── results/
│   ├── accuracy_report.md
│   └── sample_outputs.json
└── notebooks/
    └── exploration.ipynb (optional)
```
