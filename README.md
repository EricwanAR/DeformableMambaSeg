# CCSeg: Costal Cartilage Segmentation Benchmark Dataset

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-CC%204.0%20BY--SA-lightgrey.svg)](https://creativecommons.org/licenses/by-sa/4.0/)

We are excited to introduce **CCSeg** — the first publicly available benchmark dataset for **Costal Cartilage Segmentation**, designed to advance research in computer-aided diagnosis and surgical systems.

## 💡 Why CCSeg?

Costal cartilage segmentation presents significant challenges:
- **Similar intensity values** between foreground (cartilage) and background tissues (e.g., liver, intercostal muscles)
- **Particularly difficult for adolescent patients** due to softer cartilage texture
- **Lack of public datasets and benchmarks** severely limits research progress in this field

## 📊 Dataset Highlights

✅ **165 high-quality CT scan cases**  
✅ **Precise voxel-level annotations** (covering each individual costal cartilage)  
✅ **Multi-age group data** (6-35 years old)  
✅ **Out-of-distribution (OOD) test set** (22 cases) for generalization validation  
✅ **Multi-center data collection** ensuring diversity

## 🏥 Data Source

- **Primary data**: Plastic Surgery Hospital, Chinese Academy of Medical Sciences (2014-2023)
- **External test data**: Second Hospital of Hebei Medical University (2021-2024)
- All data approved by ethics committees with informed patient consent

## 🔬 Professional Annotation Process

Segmentation was performed independently by 4 plastic surgery residents under the guidance of radiology experts, with final review and correction by senior plastic surgeons to ensure annotation accuracy and consistency.

## 📈 Dataset Split

- **Training set**: 85 cases
- **Validation set**: 40 cases
- **Test set**: 40 cases
- **OOD test set**: 22 cases

This benchmark dataset provides a solid foundation for medical image analysis research related to costal cartilage, with significant application value in plastic surgery procedures such as auricular reconstruction.

## 📥 Download & Resources

- **Dataset**: [OSF | CCSeg](https://osf.io/ccseg)
- **Paper**: [Costal cartilage segmentation with topology guided deformable mamba: Method and benchmark](https://www.sciencedirect.com/science/article/pii/S0957417425000016)
- **Code**: [GitHub - EricwanAR/DeformableMambaSeg](https://github.com/EricwanAR/DeformableMambaSeg)

## 📖 Citation

If you use this dataset or code in your research, please cite our paper:

```
@article{wang2025costal,
  title={Costal cartilage segmentation with topology guided deformable mamba: Method and benchmark},
  author={Wang, Senmao and Gong, Haifan and Cui, Runmeng and Wan, Boyao and Hu, Zhonglin and Yang, Haiqing and Zhou, Jingyang and Jiang, Haiyue and Lin, Lin},
  journal={Expert Systems with Applications},
  pages={130085},
  year={2025},
  publisher={Elsevier}
}
```  
