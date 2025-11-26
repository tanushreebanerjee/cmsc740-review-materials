# Part 5: Data-Driven Shape Modeling

## Table of Contents
1. [Deep Learning Architectures for Shape Analysis](#deep-learning-architectures)
2. [Generative Modeling for Shapes](#generative-modeling)
3. [Shape Synthesis via 2D Generative Modeling](#2d-generative-modeling)
4. [Controllable Synthesis](#controllable-synthesis)

---

## Deep Learning Architectures for Shape Analysis

### Problem Statement

**Goal:** Input 3D shapes into neural networks for analysis or modeling tasks

**Operations:**
- Categorize, classify 3D shapes
- Segment 3D shapes into parts
- Retrieval from databases
- Synthesis of new shapes
- Completion of partial data
- Interactive editing, modeling
- Multi-modal modeling (shapes, text, images)

### Challenges

1. **3D shape representation:** Which representation to use?
   - Meshes
   - Point clouds
   - Voxel grids (implicit functions, binary occupancy)
   - 2D parameterizations

2. **Network architecture design:**
   - How to operate on 3D shapes?
   - 2D surfaces embedded in 3D
   - Non-uniformly sampled

### Supervised Learning Setup

**Training data:**
- Input: 3D shapes
- Labels: Ground truth outputs (class, segmentation, description, etc.)
- Loss: Difference between network output and labels

**Objective:** Optimize network weights to minimize loss

### Neural Network Architectures for 3D Shapes

#### 1. Convolutional Networks on 3D Shapes

**Goal:** Generalize 2D convolutions to 3D surfaces

**Properties of 2D convolution:**
- Sparse, linear (efficient)
- Local (receptive field)
- Translation equivariant
- Multi-scale analysis via stacking

**3D Voxel Grids:**
- Discretize shape on 3D grid
- Perform 3D convolution
- **Advantages:** Straightforward
- **Disadvantages:** Memory overhead (3D vs 2D), not rotation invariant

**Graph Convolutional Networks:**
- Represent mesh as graph
- Convolutions on graph structure
- **Advantages:** Works directly on meshes
- **Disadvantages:** More complex implementation

#### 2. PointNet

**Key idea:** Process point clouds directly without conversion

**Architecture:**
- Point-wise MLP
- Max pooling (symmetric function for permutation invariance)
- Global feature vector

**Properties:**
- Permutation invariant
- Handles variable number of points
- Efficient

**Limitations:**
- No local structure modeling
- PointNet++ extends with hierarchical processing

#### 3. Transformers

**Self-attention mechanism:**
- Attend to all points/patches
- Learn relationships between parts
- **Advantages:** Long-range dependencies, flexible
- **Disadvantages:** Quadratic complexity in sequence length

### Practice Problem 1

**Question:** A 3D shape is discretized into a 128×128×128 voxel grid. If each voxel stores a single float (4 bytes), how much memory is required? Compare to a triangle mesh with 10,000 vertices (3 floats per vertex) and 20,000 triangles (3 integers per triangle).

**Solution:**
- Voxel grid: 128³ × 4 bytes = 2,097,152 × 4 = **8,388,608 bytes ≈ 8 MB**
- Mesh vertices: 10,000 × 3 × 4 = 120,000 bytes
- Mesh triangles: 20,000 × 3 × 4 = 240,000 bytes
- Mesh total: **360,000 bytes ≈ 0.36 MB**

The mesh is much more memory-efficient for this example!

---

## Generative Modeling for Shapes

### Problem Statement

**Goal:** Generate new 3D shapes similar to training data

**Abstract View:**
- Objects in database = samples of non-uniform probability density
- Generative model: Maps random noise z ~ p_z to data space
- Goal: Generated distribution matches data distribution

**Applications:**
- Content creation
- Data augmentation
- Shape completion
- Style transfer

### Training Objectives

**Challenge:** How to formulate loss function?

**Approaches:**
1. **Estimate densities explicitly:** Minimize divergence (KL, JS, Wasserstein)
2. **Adversarial training:** GANs (minimize divergence without explicit densities)
3. **Maximum likelihood:** VAE, flow-based models
4. **Score-based:** Estimate gradient of density, use Langevin/diffusion sampling

### Variational Autoencoders (VAEs)

**Architecture:**
- Encoder: Shape → latent code z (with uncertainty)
- Decoder: Latent code z → Shape
- Latent space: Lower-dimensional representation

**Training:**
- Reconstruction loss: ||x - decode(encode(x))||²
- Regularization: KL divergence to prior (typically N(0,I))
- **ELBO (Evidence Lower BOund):** Maximize log p(x) ≥ E[log p(x|z)] - KL(q(z|x) || p(z))

**Sampling:**
- Sample z ~ N(0,I)
- Decode to generate new shape

### Generative Adversarial Networks (GANs)

**Key Paper:** Goodfellow et al., 2014

**Architecture:**
- Generator G: z → Shape (z ~ p_z, typically N(0,I))
- Discriminator D: Shape → probability(real)

**Mathematical Formulation:**
```
min_G max_D V(D,G) = E_{x~p_data}[log D(x)] + E_{z~p_z}[log(1 - D(G(z)))]
```

**Training:**
- **Generator:** Minimize V (maximize log D(G(z)))
- **Discriminator:** Maximize V (distinguish real from fake)
- Two-player minimax game
- Switch between optimizing G and D

**Theoretical Analysis:**
- GAN training minimizes **Jensen-Shannon divergence** between p_G and p_data
- Optimal discriminator: D*(x) = p_data(x) / (p_data(x) + p_G(x))
- At optimum: p_G = p_data, D* = 1/2

**KL Divergence:**
```
KL(P||Q) = Σ_x P(x) log(P(x)/Q(x))
```
- Measures "extra bits" when encoding P using optimal code for Q
- Not symmetric

**JS Divergence:**
```
JS(P||Q) = (1/2) KL(P||M) + (1/2) KL(Q||M)
```
Where M = (P + Q)/2
- Symmetric version of KL
- √JS is a metric

**Pros:**
- Theoretical guarantees
- Fast generation (no iteration)
- Conceptually simple

**Cons:**
- Training unstable
- Mode collapse (low diversity)
- Requires hyperparameter tuning

### Diffusion Models

**Key Paper:** Sohl-Dickstein et al., 2015; DDPM (Ho et al., 2020)

**Basic Idea:**
- Forward diffusion: Iteratively add noise to data → Gaussian
- Reverse diffusion: Learn to denoise → generate

**Forward Diffusion:**
```
q(x_t | x_{t-1}) = N(x_t; √(1-β_t) x_{t-1}, β_t I)
```

Where β_t is noise schedule (variance at step t)

**Joint distribution:**
```
q(x_{1:T} | x_0) = Π_{t=1}^T q(x_t | x_{t-1})
```

**Reverse Diffusion:**
- Learn p_θ(x_{t-1} | x_t) to reverse process
- If β_t small enough, reverse is also Gaussian
- Train network to predict mean μ_θ(x_t, t)

**Training Objective:**
- Minimize negative log-likelihood
- Practical: Train noise prediction network ε_θ(x_t, t)
- Loss: ||ε_t - ε_θ(x_t, t)||²

**Sampling:**
```
x_T ~ N(0,I)
for t = T to 1:
    x_{t-1} = (1/√(1-β_t)) (x_t - (β_t/√(1-α_t)) ε_θ(x_t, t)) + σ_t z
```

Where α_t = Π_{s=1}^t (1-β_s), z ~ N(0,I)

**Advantages:**
- Stable training
- High quality samples
- No mode collapse

**Disadvantages:**
- Slow generation (many steps)
- Requires many iterations

### Shape Representations for Generation

**Voxel grids:**
- 3D CNN generators
- **Advantages:** Simple, regular structure
- **Disadvantages:** Memory intensive, limited resolution

**Point clouds:**
- Point cloud generators
- **Advantages:** Efficient, flexible
- **Disadvantages:** No explicit connectivity

**Meshes:**
- Mesh generators
- **Advantages:** Explicit topology
- **Disadvantages:** Complex to generate

**Implicit functions:**
- SDF or occupancy networks
- **Advantages:** Continuous, watertight
- **Disadvantages:** Need isosurface extraction

### Practice Problem 2

**Question:** A VAE has a latent space of dimension d=128. If we quantize each dimension to 8 bits, how many distinct shapes can be represented in the quantized latent space?

**Solution:**
- Each dimension: 2⁸ = 256 possible values
- Total combinations: 256¹²⁸ = (2⁸)¹²⁸ = 2¹⁰²⁴
- This is an astronomically large number! (Much larger than number of atoms in universe)
- **Answer:** 2¹⁰²⁴ distinct shapes (theoretically, though many may not be valid)

---

## Shape Synthesis via 2D Generative Modeling

### Motivation

**Problem:** 3D generative models are complex and data-intensive

**Idea:** Leverage powerful 2D generative models (trained on images) for 3D shape generation

### EG3D: Efficient Geometry-aware 3D GANs

**Key Paper:** "Efficient Geometry-aware 3D Generative Adversarial Networks" (CVPR 2022)

**Goal:** Train 3D GAN without 3D supervision (only 2D images)

**Architecture:**
1. **3D Shape Generator:** z → tri-plane features
2. **Differentiable NeRF Rendering:** tri-planes → image
3. **Image GAN Discriminator:** image → real/fake

**Hybrid Representation (Tri-planes):**
- Instead of full 3D grid or pure MLP
- Three 2D feature grids (XY, XZ, YZ planes)
- For 3D point: project onto 3 planes, lookup features, average
- Feed to small MLP to predict density and radiance

**Advantages:**
- Faster training than pure MLP
- Higher resolution than full 3D grid
- Memory efficient

**StyleGAN2 Generator:**
- Well-engineered architecture
- Mapping network: z → w (latent space)
- Modulation/demodulation for style control
- Convolutional layers with learned weights

**Training:**
- Image-based discriminator (no 3D data needed)
- Camera parameters estimated from images
- Conditional on camera parameters

### DreamFusion: Text-to-3D

**Key Paper:** "DreamFusion: Text-to-3D using 2D Diffusion" (2022)

**Goal:** Generate 3D shapes from text descriptions

**Approach:**
- Leverage pre-trained text-to-image diffusion model
- Use Score Distillation Sampling (SDS)
- Differentiable rendering connects 3D to 2D

**Score Distillation Sampling:**
- Render 3D shape from random viewpoint
- Evaluate diffusion model on rendered image
- Use gradient to update 3D representation
- Diffusion model frozen (not trained)

**Key insight:** Diffusion model provides "score" (gradient of log density) that guides 3D optimization

### Approaches

#### 1. Multi-View Generation

**Process:**
1. Generate multiple 2D views using 2D GAN
2. Reconstruct 3D shape from views
3. Ensure multi-view consistency

**Advantages:**
- Leverage large 2D image datasets
- Powerful 2D generators available

**Challenges:**
- Multi-view consistency
- View-dependent vs. view-independent features

#### 2. 2D Parameterization

**Process:**
1. Flatten 3D shape to 2D domain (UV mapping)
2. Generate 2D texture/geometry maps
3. Map back to 3D

**Advantages:**
- Direct use of 2D generators
- Can generate texture and geometry

**Challenges:**
- Parameterization quality
- Distortion in mapping

#### 3. Image-to-3D

**Process:**
1. Generate 2D image (using 2D GAN)
2. Predict 3D shape from image (image-to-3D network)

**Advantages:**
- End-to-end training possible
- Can condition on text/images

**Challenges:**
- Ambiguity (multiple 3D shapes for one image)
- Need paired training data

### Practice Problem 3

**Question:** A 3D shape is parameterized to a 512×512 2D texture map. If we generate this map using a 2D GAN trained on natural images, what challenges might we face?

**Solution:**
**Challenges:**
1. **Semantic mismatch:** Natural images vs. geometry/texture maps
2. **Distortion:** UV mapping may introduce distortion
3. **Seam continuity:** Texture map edges must match (seamless)
4. **Geometry vs. appearance:** Need to separate geometry and texture
5. **Resolution:** 512×512 may not capture fine details

---

## Controllable Synthesis

### Problem Statement

**Goal:** Generate shapes with specific properties or constraints

**Control mechanisms:**
- Class labels (chair, table, etc.)
- Text descriptions
- Partial shapes (completion)
- Style transfer
- Semantic attributes (size, color, etc.)

### Approaches

#### 1. Conditional Generation

**Conditional VAE/GAN:**
- Input: Condition c (label, text, etc.) + latent code z
- Output: Shape conditioned on c

**Training:**
- Paired data: (shape, condition)
- Learn p(shape | condition)

#### 2. Style Transfer

**Process:**
1. Extract style from reference shape
2. Apply to content shape
3. Generate new shape with transferred style

**Methods:**
- Feature space manipulation
- Adversarial training
- Neural style transfer

#### 3. Shape Editing

**Interactive editing:**
- User provides constraints (points, curves, regions)
- Network generates shape satisfying constraints

**Methods:**
- Constrained optimization
- Latent space manipulation
- Direct shape manipulation with learned priors

#### 4. Text-to-Shape

**Process:**
1. Encode text description
2. Generate shape from text encoding
3. Ensure semantic consistency

**Challenges:**
- Text-shape alignment
- Ambiguity in descriptions
- Multi-modal training data

### Practice Problem 4

**Question:** A conditional VAE generates chairs. The latent space has dimension 64, and the condition (chair type: "office", "dining", "armchair") is one-hot encoded (3 dimensions). What is the total input dimension to the decoder?

**Solution:**
- Latent code z: 64 dimensions
- Condition c: 3 dimensions (one-hot)
- Total input: 64 + 3 = **67 dimensions**

The decoder takes this 67-dimensional vector and outputs a 3D shape representation.

---

## Summary: Part 5 Key Takeaways

1. **Deep learning architectures** for 3D shapes include CNNs, PointNet, and Transformers
2. **Generative modeling** uses VAEs, GANs, or diffusion models to create new shapes
3. **2D generative modeling** can be leveraged for 3D shape synthesis via multi-view or parameterization
4. **Controllable synthesis** enables generation with specific properties or constraints
5. Choice of **shape representation** (voxels, points, meshes, implicit) affects architecture and capabilities

---

**Previous:** [Part 4: 3D Geometry Processing & Scene Reconstruction](review_part4_geometry_reconstruction.md) | [Back to Index](REVIEW_INDEX.md)

