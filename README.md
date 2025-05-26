# Eye Blink Data Analysis  

This repository contains the **official code implementation** for the conference paper:

**[Fast sampling electrooculogram (EOG) for recording blinking kinematics](https://iovs.arvojournals.org/article.aspx?articleid=2799795)**  
📄 *Presented at ARVO 2024*
---

## 📌 Overview

This work presents a high-temporal-resolution approach for capturing and analyzing human eye blinking behavior using fast-sampling **electrooculogram (EOG)** signals.

Key contributions include:

- A **custom signal processing pipeline** for detecting and analyzing blink events from raw EOG traces
- Characterization of **blink kinematics** such as onset latency, duration, peak velocity, and recovery
- Annotations and visualizations of blink events aligned with physiological parameters
- Evaluation of EOG signals recorded at **high sampling rates** (1000 Hz) for enhanced temporal precision

> 🎯 This dataset and pipeline are valuable for researchers studying **oculomotor function**, **neurological health**, and **fatigue detection** in both clinical and experimental settings.

---

## 📚 Dataset

The original analysis was performed on a **proprietary dataset** collected under IRB-approved clinical protocols. Due to privacy and ethical considerations, this dataset **cannot be shared publicly**.

However, to allow users to explore and test the pipeline:

- A **simulated EOG test dataset** is included in the repository under:
  ```
  systane/data/
  ```

This sample data mimics the structure and characteristics of real EOG recordings and can be used to:

- Test the blink detection pipeline
- Validate visualization outputs
- Understand the preprocessing workflow

> 📌 For applying the pipeline to your own data, ensure the signals follow the same sampling frequency and format conventions as shown in the provided examples.

---

## 🛠️ Requirements

The code was developed and tested using **Python 3.9+**. Below are the core dependencies required to run the blink detection and analysis pipeline:

### 📦 Core Libraries

- `pandas`  
- `scikit-learn`  
- `opencv-python`

### 🧠 Signal Processing

- `scipy`  
- `numpy`  
- `matplotlib`  
- `biosppy` *(for physiological signal analysis, optional)*  
- `pywt` *(for wavelet transforms, optional)*

### ✅ Installation

You can install the required packages with:

```bash
pip install pandas scikit-learn opencv-python scipy numpy matplotlib biosppy pywt
```

> 💡 We recommend using a virtual environment (e.g., `venv` or `conda`) to manage dependencies.

---

## 📁 Repository Structure

Below is an overview of the main components of this repository:

```
├── blinked_working_directory.ipynb   # Main notebook for exploring and testing the pipeline
├── blinked/                          # Custom blink detection and signal analysis package
├── blinked_master/                  # Optimized version of the blink analysis pipeline
├── Systane/
│   ├── data/test_data.txt              # Simulated test EOG data
│   └── Read_Data.ipynb                 # Script to load and parse the sample test data
├── blink_params.py                  # Helper utility functions for blink detection (editable)
├── settings.py                      # Stores the path to the input data
```

> ⚠️ **Important:**  
> If you are using the **test dataset**, make sure to **update the data path** in `settings.py` accordingly.


---

## 📝 Citation

If you use this code or methodology in your research, please cite the following:

```bibtex
@article{
  title={Fast sampling electrooculogram (EOG) for recording blinking kinematics},  
  author={Sangly P Srinivas; Roselin Kiruba; Sudhir RR; Geetha K Iyer; Chetana Krishnan; Tapan Ravi; Prema Padmanabhan},
  conference={The Association for Research in Vision and Ophthalmology},
  journal={Investigative Ophthalmology & Visual Science},  
  year={2024},  
  volume={65(7)},
  page={6587},  
  url={[https://ejournal.um.edu.my/index.php/MJCS/article/view/35825](https://iovs.arvojournals.org/article.aspx?articleid=2799795)}  
}
```

---

## 📄 License

This project is licensed under the **MIT License**.

