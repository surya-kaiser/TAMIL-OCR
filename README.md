RTSBA-Based Tamil Character Recognition Using Deep Learning

## Abstract

This project focuses on recognizing handwritten Tamil characters using a deep learning model based on the RTSBA architecture. Tamil script contains complex curved shapes, loops, and diverse writing styles, making OCR significantly challenging. The RTSBA model improves recognition accuracy by combining residual CNN blocks with attention mechanisms and bottleneck layers to handle curve variations, pen pressure differences, and noise in digital handwritten input. The model is trained using a structured Tamil dataset and evaluated through accuracy, precision, recall, and F1-score. With a recognition performance exceeding 98%, the system significantly outperforms traditional OCR tools like Google Docs, Unicode OCR, and SUBASA OCR. This project is suitable for handwriting digitization, digital pads, document analysis, and Tamil-language AI applications.

---

## Team Members

* Shakthisurya (23MIA1151)
* Shreenidhi (23MIA1080)
* Bodhana (23MIA1030)

---

## Base Paper Reference

Tamil OCR Conversion from Digital Writing Pad: Recognition Accuracy Improves through Modified Deep Learning Architectures
V. Jayanthi & S. Thenmalar
Journal of Sensors, 2023* fileciteturn1file0

---

## Tools & Libraries Used

* Python 3.x
* PyTorch
* torchvision
* PaddleOCR (for comparison)
* OpenCV
* NumPy
* Matplotlib
* ReportLab (for diagrams)
* Google Colab / Jupyter Notebook

---

## Dataset Description

* Tamil handwritten images (single characters)
* Images resized to **48×200** or **32×128** based on model
* Contains **Tamil consonants, vowels, and combined forms**
* Includes curve-heavy characters, open curves, closed curves
* Supports multiple writing styles and dialect variations
* Used for training RTSBA and CRNN baselines

---

## Steps to Execute the Code

### 1. Training (RTSBA Model)

```bash
python train_rtsba_tamil.py --data dataset/ --epochs 50 --batch_size 32
```

### 2. Testing a Single Image

```bash
python test1.py --img "path/to/image.png"
```

### 3. Export Inference Model (PaddleOCR)

```bash
python tools/export_model.py -c configs/rec/rec_tamil.yml -o Global.pretrained_model=./output_tamil/iter_epoch_20
```

### 4. Run OCR Prediction

```bash
python tools/infer/predict_rec.py --image_dir sample.png --rec_model_dir inference_tamil
```



## Output Summary / Result Screenshots

* Model achieved **98.7% accuracy** on custom Tamil dataset
* Inference real-time on GPU



