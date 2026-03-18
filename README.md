# Autoencoder-Video-Restoration-Pipeline
### By Javier Fernández Ramos, María Ángeles Muñoz Juan-Dalac and Marta González Pérez

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![Status](https://img.shields.io/badge/Status-Resource--Optimized-success.svg)

## Project Overview
This project implements a **Convolutional Autoencoder** designed to restore and enhance degraded video sequences. By leveraging high-quality footage from the **2025 Monaco Grand Prix**, I developed a paired-dataset training strategy to teach the model how to reverse complex visual degradation (pixelation, blur, and color loss).

The system doesn't just process frames; it reconstructs enhanced videos while maintaining **temporal consistency** and re-integrating original audio.

## Resource-Efficient AI
A key highlight of this project is its **resilience to hardware constraints**. Due to RAM limitations on cloud platforms, the entire pipeline was optimized for **local CPU execution**:
- **Optimized `tf.data.Dataset` Pipeline:** Built to handle frame loading and shuffling with high throughput.
- **Memory Management:** Implemented custom generators and small batch sizes to ensure stability during training without sacrificing reconstruction quality.

## Methodology & Data Engineering
### Synthetic Degradation Suite
To train the model without a pre-existing paired dataset, I built a degradation engine that simulates:
1. **Resolution Loss:** 1/3 downscaling followed by upscaling to introduce realistic pixelation.
2. **Optical Blur:** Controlled Gaussian blur to simulate low-quality lenses.
3. **Chromatic Washout:** Partial desaturation to mimic aged or low-end sensor footage.

### Model Architecture
- **Encoder:** Convolutional layers with strided downsampling for hierarchical feature extraction.
- **Decoder:** Transposed convolutions and resizing layers for high-fidelity reconstruction.
- **Activation:** ReLU for hidden layers and Sigmoid for normalized pixel output.

## Results
The model was validated using a real-world test case (`moon_salute.mp4`), achieving:
- **Visual Clarity:** Significant reduction in pixelation and noise.
- **Color Recovery:** Restored vibrancy in previously washed-out frames.
- **Audio Integration:** Final output includes automated audio-video merging to deliver a complete media file.
