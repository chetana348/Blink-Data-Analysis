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


# Citation
If you use this code for your work, please cite us. 
```
@article{
  title={Real-Time Eye Tracking Using Heat Maps},  
  author={Sangly P Srinivas, Roselin Kiruba, Sudhir RR, Geetha K Iyer, Chetana Krishnan, Tapan Ravi, Prema Padmanabhan},
  conference={The Association for Research in Vision and Ophthalmology},
  journal={Investigative Ophthalmology & Visual Science},  
  year={2024},  
  volume={35(4)},
  page={339–358},
  doi = {https://doi.org/10.22452/mjcs.vol35no4.3},  
  url={https://ejournal.um.edu.my/index.php/MJCS/article/view/35825}  
}
```

# License
This project is licensed under the MIT License.
