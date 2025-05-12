# Deepfake Detection in DeepfakesAdvTrack - Spring 2025
This is the offical code for deepfake generation in DeepfakesAdvTrack, the practice session of the course Artificial Intelligence Security, Attacks and Defenses (UCAS, spring 2025).

## ⚡ How to start quickly
1. Download the code base
```
git clone https://github.com/UCASAISecAdv/DeepfakesAdvTrack-Spring2025.git
cd DeepfakesAdvTrack-Spring2025/detection
```

2. Prepare the running environment
```
conda create -n course_AISA python==3.7.12
conda activate course_AISA
pip3 install --upgrade pip
pip install -r requirements.txt
```

3. Download the dataset

Please acquire the download link from our Wechat.
- Training set: CelebDF-v2-training ([access require](https://github.com/yuezunli/celeb-deepfakeforensics)). The use of CelebDF-v1/v2 val/test set is forbidden. If you use the data from other sources for training, you should declare it in your final course report.
- Validation set: UCAS_AISA-val. The labels are included.
- Test set 1: UCAS_AISA-test1. No label is given.
- Test set 2: UCAS_AISA-test2. It includes the images collected from Deepfake Generation and will be released at the last week of the practice.

4. Start model inference
```
CUDA_VISIBLE_DEVICES=0 python inference.py \
    --your-team-name $YOUR_TEAM_NAME \
    --data-folder $YOUR_DATASET_PATH/test1 \
    --model-weights ./utils/weights.ckpt \
    --result-path $YOUR_SAVE_PATH
```

5. Start model evaluation

We evaluate a model according to AUC. Please refer to the corresponding file.
```
python evaluate.py \
    --submit-path ${YOUR_SAVE_PATH}/${YOUR_TEAM_NAME}
```

## ⚠️ Caution
1. You can customize your transforms and models in `inference.py`. **DO NOT** modify any other codes, otherwise the way your results are calculated may be affected and your rating will be incorrect.
2. Fake is mapped to 1 and Real is mapped to 0.

## Acknowledgement
This code is based on [DFGC_startkit](https://github.com/bomb2peng/DFGC_starterkit/tree/master).
The source data is [CelebDF](https://github.com/yuezunli/celeb-deepfakeforensics).