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

**Applications:**
- Content creation
- Data augmentation
- Shape completion
- Style transfer

### Approaches

#### 1. Variational Autoencoders (VAEs)

**Architecture:**
- Encoder: Shape → latent code z
- Decoder: Latent code z → Shape
- Latent space: Lower-dimensional representation

**Training:**
- Reconstruction loss: ||x - decode(encode(x))||²
- Regularization: KL divergence to prior (typically N(0,I))

**Sampling:**
- Sample z ~ N(0,I)
- Decode to generate new shape

#### 2. Generative Adversarial Networks (GANs)

**Architecture:**
- Generator: z → Shape
- Discriminator: Shape → Real/Fake probability

**Training:**
- Adversarial: Generator tries to fool discriminator
- Discriminator tries to distinguish real from fake

**Sampling:**
- Sample z ~ N(0,I)
- Generate shape using generator

#### 3. Diffusion Models

**Process:**
- Forward: Gradually add noise to shape
- Reverse: Learn to denoise (generate)

**Training:**
- Learn denoising network
- Sample by reversing diffusion process

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

