
This is an implementation of the **Convolutional Block Attention Module (CBAM)** paper for the **PIDNet**, developed as part of a university project for a forest fire segmentation challenge.

Two main branches are available:

* `pidnet-baseline`: The original PIDNet without attention.
* `pidnet-cbam`: PIDNet with CBAM integrated for improved segmentation performance.

The models are trained on a combination of:

* **Synthetic Drone Fire Dataset** 
* **Corsican Fire Database** 
(provided by the challenge)

---

## Requirements

You need to install the required Python packages:

```bash
pip install -r requirements.txt
```

---

## Training

Example training command (for a specific seed):

```bash
python tools/train.py --cfg configs/fire/pidnet_medium_camvid_flame.yaml --seed 0
```

Trained models are saved under `output/fire/<branch>/seed<SEED>/`.

---

## Evaluation

Evaluate a trained model using:

```bash
python tools/val.py --cfg configs/fire/pidnet_medium_camvid_flame.yaml --weight output/fire/cbam/seed0/best.pt
```

Make sure the `MODEL_FILE` in your YAML config is correctly set to the corresponding `.pt` file.

---

## More Details

For a full explanation of the experiments, CBAM integration, and performance comparisons between the baseline and the CBAM-enhanced model, please refer to the accompanying report PDF in the repository.
[Project Presentation](https://github.com/iogisdaki/fireSegmentationChallenge/blob/pidnet_baseline/PIDNet/docs/PIDNet_CBAM_Presentation.pdf)

---

## Acknowledgments

This work is based on:

* [PIDNet (CVPR 2023)](https://arxiv.org/abs/2206.02066)
* [CBAM: Convolutional Block Attention Module (ECCV 2018)](https://arxiv.org/abs/1807.06521)
