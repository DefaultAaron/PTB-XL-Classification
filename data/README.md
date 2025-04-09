# Data Directory

This directory contains the PTB-XL ECG dataset files used in this project.

## Structure

- `raw/`: Contains the original PTB-XL dataset files (subset, excluding wavelet data)
- `processed/`: Directory for storing preprocessed data (not included in repository)

## Dataset Information

The data in this repository is derived from the PTB-XL dataset, a large publicly available electrocardiography dataset developed by the Physikalisch-Technische Bundesanstalt (PTB).

### Source
- Original dataset: [PTB-XL on PhysioNet](https://physionet.org/content/ptb-xl/)

### Important Notes
1. **Incomplete Dataset**: This repository contains only a subset of the original PTB-XL dataset. The wavelet data from the 'records100' and 'records500' folders has been omitted due to GitHub storage limitations.
2. **Missing Files**: To work with the complete dataset, please download the missing files from the original source.
3. **Processed Data**: When you run the preprocessing notebooks, the processed data will be saved in the `processed/` directory (which is not committed to the repository).

### Usage
To properly use this dataset:
1. Clone this repository
2. If you need the complete dataset, download the missing wavelet data from the original source and place it in the appropriate folders
3. Run the preprocessing notebooks to generate the processed data

### Citation
If you use this data in academic work, please cite the original paper:

Wagner, P., Strodthoff, N., Bousseljot, R., Kreiseler, D., Lunze, F. I., Samek, W., & Schaeffter, T. (2020). PTB-XL, a large publicly available electrocardiography dataset. Scientific Data, 7(1), 1-15.