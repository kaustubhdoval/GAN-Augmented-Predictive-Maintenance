# Generating Synthetic Failure Data

**DATASET IS CONFIDENTIAL AND HAS NOT BEEN UPLOADED TO GITHUB**

In real-world applications - such as industrial machinery operating on a factory floor - failure data is inherently scarce. Training models exclusively on nominal (“healthy”) data results in significant class imbalance, which can adversely affect model performance and reliability.

To address this challenge, we employ Generative Adversarial Networks (GANs) to synthesize representative failure data. This approach mitigates class imbalance and enables the development of more robust and accurate predictive models. You can refer to the section on GANs to learn more about them.

## Dataset

Tri-axial data from a CNC Machine

## Approach

I decided to utilize a **C-WGAN-GP** ie. Conditional Wasserstein Generative Adversarial Networks with Gradient Penalty primarily for its simple usage, superior ability to recognize and capture local oscillation patterns (like chatter) and for enforcing the Lipschitz constraint - data doesn't deviate excessively from training data.

## Pipeline

```
Raw CSVs (X, Y, Z signals)
        │
        ▼
┌─────────────────────────────┐
│  load_csv_to_windows()      │  Overlapping windows (size=200, overlap=50%)
│  Per-file, vectorised       │  → (N, 3, 200) float32 array
└─────────────────────────────┘
        │
        ▼
┌─────────────────────────────┐
│  per_window_normalise()     │  Each window independently scaled to [-1, 1]
│                             │  Saves (w_min, w_range) for denormalisation
└─────────────────────────────┘
        │
        ▼
┌─────────────────────────────┐
│  Balanced subsample         │  5000 chatter + 5000 no-chatter (configurable)
│  build_dataloader()         │  Shuffled, pinned, persistent workers
└─────────────────────────────┘
        │
        ▼
┌─────────────────────────────┐
│  train()                    │  Conditional WGAN-GP
│  - N_CRITIC steps / batch   │  Critic updated N_CRITIC × per generator step
│  - Instance noise (decay)   │  σ: 0.05 → 0.0 over first 20% of epochs
│  - Cosine LR annealing      │  Both G and C schedulers
│  - Early stopping           │  Patience-based on W-distance
└─────────────────────────────┘
        │
        ▼
┌─────────────────────────────┐
│  generate_windows()         │  Conditioned on label + RPM class
│  save_windows_to_csv()      │  → dataset/synthetic/{rpm}_{label}_synth_NNNN.csv
└─────────────────────────────┘
```

---

## Architecture

The model is a **Conditional WGAN-GP** operating on raw 1-D tri-axial time series.
Conditioning is a concatenation of a 2-class label one-hot and a 6-class RPM one-hot → **8-dimensional condition vector**.

### Generator

```
Noise (128,) + Condition (8,)
        │
        ▼  Linear projection
(256, 25)       ← start = WINDOW_SIZE // 2^(n_upsample) = 200 // 8 = 25
        │
        ▼  × 3 ConvTranspose1d blocks (stride=2, doubles length each time)
        │   Condition injected via channel-concat before each block
(128, 50) → (64, 100) → (32, 200)
        │
        ▼  ResBlock1D
        │   Condition injected at block INPUT (before BatchNorm)
        │   Skip connection via 1×1 Conv
(32, 200)
        │
        ▼  Conv1d(32 + 8, 3, kernel=1) + Tanh
Output: (3, 200)   ← (N_CHANNELS, WINDOW_SIZE)
```

### Critic

```
Input: (3, 200) + Condition (8,)
        │   Condition broadcast → (8, 200), concat on channel dim
        ▼
(11, 200)
        │
        ▼  × 4 Conv1d blocks (stride=2, halves length each time)
        │   Condition re-injected between each block
        │   No BatchNorm (destabilises gradient penalty)
(32, 100) → (64, 50) → (128, 25) → (256, 12)
        │
        ▼  AdaptiveAvgPool1d(1) → flatten
        ▼  Linear(256, 1)
Output: scalar Wasserstein score
```

### Training Objective

```
Critic loss  = E[C(fake)] − E[C(real)] + λ·GP        λ = 10
Generator loss = −E[C(fake)]
GP           = gradient penalty on random interpolations (real ↔ fake)
Instance noise decays linearly: σ = 0.05 × max(0, 1 − epoch/noise_decay_end)
```

## About GANs

### Structure

1. **Generator** : Learns to generate plausible real data
2. **Discriminator**: Learns to distinguish between generators fake and real data (like a classifier). It also penalizes the generator for producing implausible results.

### Training the Discriminator

- Connects to two loss functions
- LOOP:
  i. Classify both real data and synthetic data (from generator)
  ii. Loss functions penalize the discriminator for mis-classification
  iii. Discriminator learns (updates weights) through backpropogation via the network

### Generator

- Every neural network needs an input, for our purpose we use noise or random input as input to the generator which then creates a synthetic data sample. (allows us to create a wide variety of samples)

- Use the discriminator to train the generator:
  1.  Sample random noise.
  2.  Produce generator output from sampled random noise.
  3.  Get discriminator "Real" or "Fake" classification for generator output.
  4.  Calculate loss from discriminator classification.
  5.  Backpropagate through both the discriminator and generator to obtain gradients.
  6.  Use gradients to change only the generator weights.

Because a GAN contains two separately trained networks, its training algorithm must address two complications:

- GANs must juggle two different kinds of training (generator and discriminator).
- GAN convergence is hard to identify.

- To solve this -> We alternate between training the Generator and Discriminator for one or more epochs

### Convergence

As the generator improves with training, the discriminator performance gets worse because the discriminator can't easily tell the difference between real and fake. If the generator succeeds perfectly, then the discriminator has a 50% accuracy. In effect, the discriminator flips a coin to make its prediction.

This progression poses a problem for convergence of the GAN as a whole: the discriminator feedback gets less meaningful over time. If the GAN continues training past the point when the discriminator is giving completely random feedback, then the generator starts to train on junk feedback and its own quality may collapse.

## References

https://www.sciencedirect.com/science/article/pii/S0019057821006169
https://en.wikipedia.org/wiki/Generative_adversarial_network
https://www.ibm.com/think/topics/generative-adversarial-networks
https://developers.google.com/machine-learning/gan/gan_structure
https://medium.com/the-research-nest/how-to-program-a-simple-gan-559ad707e201
