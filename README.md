
## Requirements 
### Installation
Follow these steps to set up a conda environment and ensure all necessary packages are installed:

```bash
git clone https://github.com/kdiAAA/TDA.git
cd TDA

conda create -n tda python=3.7
conda activate tda

# The results are produced with PyTorch 1.12.1 and CUDA 11.3
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch

pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1+cu116 -f https://download.pytorch.org/whl/torch_stable.html

pip install -r requirements.txt
```

### Dataset
To set up all required datasets, kindly refer to the guidance in [DATASETS.md](docs/DATASETS.md), which incorporates steps for two benchmarks.

## Run TDA
### Configs
The configuration for TDA hyperparameters in `configs/dataset.yaml` can be tailored within the provided file to meet the needs of various datasets. This customization includes settings for both the positive and negative caches as outlined below:
* **Positive Cache Configuration:** Adjustments can be made to the `shot_capacity`, `alpha`, and `beta` values to optimize performance.

* **Negative Cache Configuration:** Similar to the positive cache, the negative cache can also be fine-tuned by modifying the `shot_capacity`, `alpha`, `beta`, as well as the `entropy_threshold` and `mask_threshold` parameters.

For ease of reference, the configurations provided aim to achieve optimal performance across datasets on two benchmarks, consistent with the results documented in our paper. However, specific tuning of these parameters for negative cache could potentially unlock further enhancements in performance. Adjusting parameters like `alpha` and `beta` within the positive cache lets you fine-tune things to match the unique needs of each dataset.

### Running
To execute the TDA, navigate to the `scripts` directory, where you'll find 4 bash scripts available. Each script is designed to apply the method to two benchmarks, utilizing either the ResNet50 or ViT/B-16 as the backbone architecture. The scripts process the datasets sequentially, as indicated by the order divided by '/' in the script. WandB logging is activated by default. If you wish to deactivate this feature, simply omit the `--wandb-log` argument. 

Below are instructions for running TDA on both Out-of-Distribution (OOD) and Cross-Domain benchmarks using various backbone architectures. Follow the steps suited to your specific needs:"

#### OOD Benchmark
* **ResNet50**: Run TDA on the OOD Benchmark using the ResNet50 model:
```
bash ./scripts/run_ood_benchmark_rn50.sh 
```
* **ViT/B-16**: Run TDA on the OOD Benchmark using the ViT/B-16 model.
```
bash ./scripts/run_ood_benchmark_vit.sh 
```

#### Cross-Domain Benchmark
* **ResNet50**: Run TDA on the Cross-Domain Benchmark using the ResNet50 model:
```
bash ./scripts/run_cd_benchmark_rn50.sh 
```
* **ViT/B-16**: Run TDA on the Cross-Domain Benchmark using the ViT/B-16 model.
```
bash ./scripts/run_cd_benchmark_vit.sh 
```




