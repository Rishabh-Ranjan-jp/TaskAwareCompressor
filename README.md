# TaskAwareCompressor

**Deep Learning-Based Semantic Image Compression and Reconstruction**

---

## Overview

This project explores a **task-aware deep learning framework for image compression**, designed to retain **semantic information** necessary for downstream vision tasks such as classification. Traditional compression techniques like JPEG focus on pixel-level fidelity, often ignoring higher-level features important for machine perception. Here, we combine a **pre-trained ResNet-18 classifier** with a custom **decoder network** to create a latent representation of images, reconstruct them, and preserve classification-relevant information.

The resulting system produces **compressed representations that are smaller in file size** than the original while retaining critical features for high classification accuracy. This approach is particularly suited for scenarios where **machine interpretation matters more than perfect visual fidelity**.

---

## Features

- **Binary Classification Focus:** Cat vs. Dog images (CIFAR-10) for proof-of-concept.
- **ResNet-18 Based Encoder:** Leverages pre-trained ImageNet weights for semantic extraction.
- **Custom Decoder:** Upsamples latent features to reconstruct images, trained with a **joint loss** to preserve both visual fidelity and classification accuracy.
- **Task-Aware Loss Function:**  
  `L_total = L_CE + ALPHA * L_MSE`  
  - `L_CE` ensures semantic information is preserved.  
  - `L_MSE` encourages visual reconstruction quality.
- **Comprehensive Evaluation:** Compare reconstructed images against originals and size-matched JPEGs for classification accuracy.
- **Modular Architecture:** Encoder, classifier head, and decoder defined as separate PyTorch modules for easy experimentation.

---

## Getting Started

### Requirements

```text
# TODO: Add actual requirements with versions here
````

### Dataset

This project uses **CIFAR-10**, filtered for the binary classes:

* **Cat:** class 3 → label 1
* **Dog:** class 5 → label 0

Images are resized to `32x32` and normalized using CIFAR-10 statistics.

### Project Structure

```text
# TODO: Update this structure with actual files and directories
```

---

## Results

* Reconstructed images preserve **semantic content** significantly better than size-matched JPEGs.
* The decoder effectively **denoises latent representations**, sometimes improving classifier performance over raw latent outputs.
* Two variants of the decoder were tested:

  * **Tanh**: slightly better compression efficiency
  * **Sigmoid**: slightly better semantic preservation

---

## Future Work

* Explore **different datasets** and higher-resolution images.
* Test **advanced decoder architectures** (residual connections, attention layers).
* Investigate **different weighting strategies for ALPHA** in joint loss.
* Experiment with **multi-class classification** beyond binary.

---

## License

This project is licensed under the **GNU General Public License v3.0 (GPLv3)**.

This means you are free to:

* Use, copy, modify, and distribute this code for any purpose.
* Distribute derivative works **under the same license**, ensuring that modifications also remain open-source.

**Disclaimer:** This project is experimental. While the code has been tested on CIFAR-10, performance may vary for other datasets or resolutions.

For the full license text, see [LICENSE](./LICENSE).

---

## Acknowledgments

* Original ResNet-18 implementation from [PyTorch torchvision](https://pytorch.org/vision/stable/models.html).
* CIFAR-10 dataset from [https://www.cs.toronto.edu/~kriz/cifar.html](https://www.cs.toronto.edu/~kriz/cifar.html).
* Inspiration from task-aware compression and deep learning autoencoder research.


