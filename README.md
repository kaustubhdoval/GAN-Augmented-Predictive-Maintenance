# GAN-Augmented Predictive Maintenance

This repository explores advanced machine learning techniques for predictive maintenance in industrial settings, with a focus on vibration-based monitoring of CNC machines. It contains two complementary applications:

## 1. Predicting Machine Status (Chatter Detection)

This project uses time-series sensor data and an XGBoost classifier to detect chatter (undesirable vibrations) in CNC machines. The pipeline includes feature engineering, normalization, and robust model training to distinguish between healthy and faulty machine states. See the [PredictingMachineStatus/README.md](PredictingMachineStatus/README.md) for details on the approach, features, and outputs.

## 2. Generating Synthetic Failure Data (GANs)

This project addresses the challenge of scarce failure data by using Generative Adversarial Networks (GANs) to synthesize realistic failure scenarios. This helps mitigate class imbalance and enables more robust predictive models. The approach leverages conditional Wasserstein GANs with gradient penalty for high-quality data generation. See the [GeneratingFailureData/README.md](GeneratingFailureData/README.md) for methodology and background on GANs in industrial applications.

---

Together, these projects demonstrate how combining traditional machine learning with modern generative models can improve predictive maintenance, especially in data-scarce industrial environments.
