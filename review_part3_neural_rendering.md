# Part 3: Neural Rendering

## Table of Contents
1. [Denoising Using Neural Networks](#denoising)
2. [Neural Radiosity](#neural-radiosity)
3. [Learning to Importance Sample in Primary Sample Space](#neural-importance-sampling)

---

## Denoising Using Neural Networks

### Problem Statement

**Challenge:** Monte Carlo rendering produces noisy images, especially with few samples per pixel.

**Solution:** Use neural networks to denoise noisy Monte Carlo renders.

### Approach

**Input:** Noisy rendered image (low sample count)
**Output:** Clean, denoised image

**Training:**
- Input: Noisy images (few samples)
- Ground truth: Reference images (many samples)
- Loss: L2 or L1 loss between denoised output and ground truth

### Key Concepts

**Neural Network Architecture:**
- Typically convolutional neural networks (CNNs)
- Input: Noisy image + auxiliary buffers (normals, depth, albedo)
- Output: Denoised RGB image

**Auxiliary Buffers (AOVs - Arbitrary Output Variables):**
- Surface normals
- Depth
- Albedo (diffuse color)
- Direct/indirect lighting separation
- Help network understand scene structure

### Advantages

1. **Fast:** Denoising faster than rendering more samples
2. **Effective:** Can reduce noise significantly
3. **General:** Works across different scenes

### Limitations

1. **Training data:** Needs diverse training scenes
2. **Artifacts:** May introduce blurring or incorrect details
3. **Generalization:** May not work well on scenes very different from training data

### Practice Problem 1

**Question:** A Monte Carlo renderer produces images with variance σ² = 100. A neural denoiser reduces this to σ² = 4. How many additional samples would be needed to achieve the same variance reduction without denoising?

**Solution:**
- Original variance: σ² = 100
- Denoised variance: σ² = 4
- Variance reduction factor: 100/4 = 25
- For Monte Carlo: variance ∝ 1/N
- To reduce variance by factor 25, need 25× more samples
- **Answer:** 25× more samples needed

---

## Neural Radiosity

### Background: Traditional Radiosity

**Radiosity:** Solution method for rendering equation restricted to **diffuse surfaces only**.

**Key simplification:**
- Radiosity B(x) doesn't depend on direction (unlike radiance L(x,ω))
- Only valid for perfectly diffuse (Lambertian) surfaces

**Radiosity equation (3-point form):**
```
B(x) = E(x) + ρ(x) ∫_M B(y) G(x,y) dA(y)
```

Where:
- B(x): Radiosity at point x
- E(x): Emitted radiosity
- ρ(x): Diffuse albedo (reflectance)
- G(x,y): Geometry term (form factor)

### Discretization Approach

**Traditional method:**
- Discretize scene into mesh faces
- Solve linear system: **B = E + F·B**
- Form factors F_i,j between faces i and j

**Challenges:**
- High mesh resolution needed for accuracy
- Very large linear systems
- Extending to arbitrary BRDFs requires 4D discretization (expensive)

### Neural Radiosity

**Key idea:** Represent radiosity B(x) using a **neural network** instead of discretization.

**Advantages:**
1. **Continuous representation:** No need for mesh discretization
2. **General BRDFs:** Can handle non-diffuse surfaces
3. **Efficient:** Neural networks can represent complex functions compactly

**Approach:**
1. Neural network takes 3D position x as input
2. Outputs radiosity B(x) (or radiance L(x,ω) for general BRDFs)
3. Train network to satisfy rendering equation

**Training:**
- Loss function: Difference between left and right side of rendering equation
- Use automatic differentiation for gradients
- Optimize network weights via gradient descent

### Neural Representations in Graphics

**General concept:** Use neural networks to represent continuous functions in graphics.

**Examples:**
- **Geometry:** Signed distance fields, occupancy functions
- **Radiance fields:** NeRF (Neural Radiance Fields)
- **Images:** Coordinate-based networks
- **BRDFs:** Neural BRDF representations

**Advantages:**
- Efficient storage (few parameters vs. dense discretization)
- Adaptive, nonlinear representations
- Leverage GPU infrastructure

**Disadvantages:**
- Slower evaluation than conventional techniques
- Requires nonlinear optimization
- Less understood convergence properties

### Practice Problem 2

**Question:** Traditional radiosity discretizes a scene into 10,000 mesh faces. If we extend to general BRDFs (requiring 4D discretization with 10 samples per dimension for direction), how many elements would we need?

**Solution:**
- 2D position: 10,000 faces
- 2D direction: 10 × 10 = 100 samples
- Total: 10,000 × 100 = **1,000,000 elements**
- This demonstrates why neural representations are attractive for general BRDFs!

---

## Learning to Importance Sample in Primary Sample Space

### Motivation

**Limitations of standard importance sampling:**
1. Samples at each bounce separately (doesn't consider full path)
2. Approximate BRDF/emitter models
3. Ignores occlusions (visibility)
4. Doesn't handle non-local effects (e.g., caustics)
5. Requires separate techniques for motion blur, depth of field

**Goal:** Learn to sample entire paths more effectively using neural networks.

### Primary Sample Space (PSS)

**Concept:**
- Paths are constructed from random numbers (canonical uniform samples)
- **Primary Sample Space:** Unit hypercube containing all random numbers
- **Path Space:** Actual 3D paths in the scene

**Mapping:**
```
y ∈ [0,1]^d  →  x = Φ(y)  (path construction via ray tracing)
```

Where:
- y: Random numbers in primary sample space
- x: Path in path space
- Φ: Path construction function (deterministic ray tracing)

### Primary Sample Space Integral

**Rendering equation in PSS:**
```
I_j = ∫_Ω f_j(x) dx = ∫_[0,1]^d f_j(Φ(y)) |det J_Φ(y)| dy
```

Where:
- f_j(x): Path contribution function
- J_Φ: Jacobian of path construction mapping
- |det J_Φ|: Change of variables factor

### Neural Importance Sampling

**Approach:**
1. Learn a transformation T: [0,1]^d → [0,1]^d that warps uniform samples
2. Sample y' ~ Uniform, then apply T to get y = T(y')
3. Construct path x = Φ(y)
4. Use importance sampling with learned PDF p_T

**Training:**
- Minimize variance of Monte Carlo estimator
- Learn transformation T using neural network
- Use automatic differentiation for gradients

### Advantages

1. **Unified approach:** Treats renderer as black box
2. **Full path consideration:** Samples entire paths, not individual bounces
3. **Handles complex effects:** Caustics, motion blur, depth of field
4. **Scene-specific:** Adapts to specific scene characteristics

### Implementation Details

**Neural network:**
- Input: Uniform random numbers
- Output: Warped random numbers
- Architecture: Typically MLP or more sophisticated (e.g., normalizing flows)

**PDF computation:**
- Need to compute p_T(y) = |det J_T(y')| where y = T(y')
- Requires computing Jacobian determinant

**Training:**
- Collect "training" samples of integrand
- Fit transformation to reduce variance
- Can be done online (during rendering) or offline

### Related Techniques

**A posteriori methods:**
- Path guiding: Cache samples, fit density
- Online learning path guiding
- Kd-tree based path guiding

**Markov Chain Monte Carlo:**
- Metropolis Light Transport (MLT)
- Primary Sample Space MLT (PSSMLT)
- Multiplexed MLT

### Practice Problem 3

**Question:** In primary sample space importance sampling, we learn a transformation T that maps uniform samples to a better distribution. If T is a simple scaling: T(y) = 2y for y ∈ [0, 0.5] and T(y) = 2y - 1 for y ∈ [0.5, 1], what is the PDF p_T(y)?

**Solution:**
- For y ∈ [0, 0.5]: T(y) = 2y, so y' = y/2
- Jacobian: dT/dy = 2, so |det J| = 2
- p_T(y) = p_uniform(y') × |det J| = 1 × 2 = 2

- For y ∈ [0.5, 1]: T(y) = 2y - 1, so y' = (y+1)/2
- Jacobian: dT/dy = 2, so |det J| = 2
- p_T(y) = 2

- **Answer:** p_T(y) = 2 for all y ∈ [0,1] (uniform distribution scaled by 2)

---

## Summary: Part 3 Key Takeaways

1. **Neural denoising** can significantly reduce noise in Monte Carlo renders
2. **Neural radiosity** uses networks to represent radiance/radiosity functions continuously
3. **Neural importance sampling** learns to sample paths in primary sample space
4. Neural representations offer **efficient, adaptive** alternatives to traditional discretization
5. All methods leverage **automatic differentiation** and GPU infrastructure

---

**Next:** [Part 4: 3D Geometry Processing & Scene Reconstruction](review_part4_geometry_reconstruction.md) | [Previous: Part 2](review_part2_advanced_rendering.md) | [Back to Index](REVIEW_INDEX.md)

