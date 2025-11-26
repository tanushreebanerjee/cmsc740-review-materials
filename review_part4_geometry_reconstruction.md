# Part 4: 3D Geometry Processing & Scene Reconstruction

## Table of Contents
1. [Introduction to 3D Geometry Processing](#introduction)
2. [3D Point Clouds and Surface Reconstruction](#point-clouds)
3. [Scene Reconstruction from Images](#scene-reconstruction)
4. [NeRF-based Avatars](#nerf-avatars)

---

## Introduction to 3D Geometry Processing

### Objectives

**Goal:** Capture, process, manipulate, and analyze 3D shapes

### Applications

- Medical imaging
- Simulation
- E-commerce
- Cultural heritage
- Scientific research
- GIS
- Architecture
- 3D printing, manufacturing
- Games, movies
- Shape databases

### Processing Pipeline

```
Acquisition → Processing → Application
```

**Acquisition:**
- 3D scanners (triangulation, time of flight)
- 3D reconstruction from images (structure from motion, multiview stereo)
- 3D imaging (microscopy, MRI, CT)
- Interactive modeling

**Processing:**
- Surface reconstruction from points
- Segmentation
- Smoothing, denoising
- Simplification

**Applications:**
- Visualization, rendering
- Manufacturing (3D printing)
- Analysis (classification, segmentation)
- Simulation
- Databases, retrieval

### 3D Shape Representations

#### 1. Point Clouds
- Array of 3D points: ((x₀,y₀,z₀), (x₁,y₁,z₁), ...)
- Often raw output of 3D scanners
- **Advantages:** Simple data structure
- **Disadvantages:** Not continuous, no topology

#### 2. Polygon Soup
- Array of separate polygons
- **Advantages:** Simple, can render
- **Disadvantages:** Not continuous, no topology, not watertight

#### 3. Polygon Meshes
- Connected polygons forming mesh
- Vertices, edges, faces, adjacencies
- **Advantages:** Continuous, has topology
- **Disadvantages:** Requires proper connectivity

#### 4. Parametric Surfaces
- B-reps (boundary representations)
- NURBS, Bézier surfaces
- **Advantages:** Smooth, continuous
- **Disadvantages:** Complex for arbitrary shapes

#### 5. Implicit Surfaces
- Defined by f(x,y,z) = 0
- Signed distance fields
- **Advantages:** Easy to combine, watertight
- **Disadvantages:** Harder to render directly

### Practice Problem 1

**Question:** A point cloud contains 1,000,000 points. If we convert it to a triangle mesh with average 6 triangles per vertex, approximately how many triangles will the mesh have?

**Solution:**
- Each triangle has 3 vertices
- Average 6 triangles per vertex means each vertex is shared by ~6 triangles
- Triangles = (vertices × triangles_per_vertex) / vertices_per_triangle
- Triangles = (1,000,000 × 6) / 3 = **2,000,000 triangles**

---

## 3D Point Clouds and Surface Reconstruction

### Surface Reconstruction Problem

**Input:** Point cloud (set of 3D points, possibly with normals)
**Output:** Continuous surface representation (mesh, implicit surface)

### Challenges

1. **Noise:** Scanned points have measurement errors
2. **Incomplete data:** Missing regions
3. **Non-uniform sampling:** Dense in some areas, sparse in others
4. **Outliers:** Incorrect measurements

### Reconstruction Methods

#### 1. Delaunay Triangulation
- Connect points to form triangles
- Maximize minimum angle (Delaunay criterion)
- **Advantages:** Guaranteed to exist, unique
- **Disadvantages:** May create unwanted triangles

#### 2. Poisson Surface Reconstruction
- Fit implicit function to point cloud
- Solve Poisson equation
- Extract isosurface (marching cubes)
- **Advantages:** Robust to noise, produces watertight surfaces
- **Disadvantages:** May fill holes incorrectly

#### 3. Moving Least Squares (MLS)
- Local surface fitting
- Weighted least squares at each point
- **Advantages:** Handles noise well
- **Disadvantages:** Computationally expensive

#### 4. RANSAC-based Methods
- Fit primitives (planes, spheres) to point subsets
- Iteratively remove inliers
- **Advantages:** Robust to outliers
- **Disadvantages:** Assumes primitive shapes

### Normal Estimation

**Problem:** Point clouds often don't include surface normals

**Methods:**
1. **PCA (Principal Component Analysis):**
   - For each point, find k nearest neighbors
   - Compute covariance matrix
   - Normal = eigenvector of smallest eigenvalue

2. **Consistent orientation:**
   - Ensure normals point outward
   - Use graph-based propagation

### Practice Problem 2

**Question:** For Poisson surface reconstruction, we solve ∇²f = ∇·V where f is the implicit function and V is a vector field. If the point cloud has n points, what is the typical computational complexity?

**Solution:**
- Poisson equation is a linear system
- System size depends on discretization (typically O(n) for n points)
- Solving linear system: O(n³) for direct methods, O(n log n) for iterative methods
- **Answer:** Typically O(n log n) with efficient solvers

---

## Scene Reconstruction from Images

### Problem Statement

**Inverse rendering problem:**
- **Input:** Images of static scene from different viewpoints + camera parameters
- **Output:** Scene parameters (geometry, materials, lighting) such that rendered images match input

**Applications:**
- Extract 3D representation
- Novel view synthesis
- Re-lighting
- Material editing

### Approach

**Optimization formulation:**
```
minimize: L = Σ_i ||I_i - f(scene, camera_i)||²
```

Where:
- I_i: Input image i
- f: Rendering function
- scene: Scene parameters (geometry, materials, etc.)
- camera_i: Camera parameters for image i

**Optimization:**
- Gradient descent
- Need gradients: ∂L/∂scene
- Use automatic differentiation

### Challenges

1. **Suitable rendering function:**
   - Must be differentiable
   - Need automatic differentiation

2. **Suitable scene parameterization:**
   - Powerful enough to reproduce input images
   - Efficient to optimize

3. **Initialization:**
   - Good starting guess needed

### NeRF (Neural Radiance Fields)

**Key paper:** "NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis" (Mildenhall et al., 2020)

**Key ideas:**
1. **Volumetric rendering** using emission and absorption
2. **Neural network** represents volume density σ(x) and radiance c(x,d)
3. **Differentiable rendering** enables optimization

### Volumetric Rendering Function

**Color of ray r(t) = o + td:**
```
C(r) = ∫_0^d T(t) σ(r(t)) c(r(t), d) dt
```

Where:
- **T(t):** Transmittance (fraction of light reaching t)
- **σ(r(t)):** Density (scattering coefficient)
- **c(r(t), d):** Radiance (volumetric emission)

**Transmittance:**
```
T(t) = exp(-∫_0^t σ(r(s)) ds)
```

### Numerical Approximation

**Ray marching:**
- Sample N points along ray at locations t_i
- Step size δ = d/N
- Approximate integral:
```
C(r) ≈ Σ_i T_i (1 - exp(-σ_i δ)) c_i
```

Where:
- T_i = exp(-Σ_{j=0}^{i-1} σ_j δ): Probability to reach segment i
- (1 - exp(-σ_i δ)): Probability of emission in segment i

### Neural Network Architecture

**Input:** 3D position x, direction d
**Output:** Density σ, radiance c

**Architecture:**
- MLP (multilayer perceptron)
- Positional encoding: γ(p) = (sin(2⁰πp), cos(2⁰πp), ..., sin(2^(L-1)πp), cos(2^(L-1)πp))
- Position: L=10, Direction: L=4

**Why positional encoding?**
- Helps network learn high-frequency details
- Without it, networks bias toward low frequencies

### Hierarchical Sampling

**Two networks:**
1. **Coarse network:** N_c = 64 samples per ray
2. **Fine network:** N_f = 128 samples per ray

**Strategy:**
- Train coarse network first
- Use coarse samples to define piecewise-constant PDF
- Sample fine network according to this PDF (importance sampling)

### Optimization

**Loss function:**
```
L = Σ_r ||C(r) - Ĉ_coarse(r)||² + ||C(r) - Ĉ_fine(r)||²
```

**Gradients:**
- Automatic differentiation (backpropagation)
- Gradients flow through rendering function to network weights

### Limitations

1. **Not physically-based:** No model of light scattering on surfaces
2. **Known cameras:** Requires known camera parameters
3. **No direct surfaces:** Volumetric representation, surfaces extracted via isosurface
4. **No re-lighting:** View-dependent radiance, not material properties

### Practice Problem 3

**Question:** In NeRF, we use positional encoding γ(p) with L=10 for positions. How many dimensions does the encoded position vector have?

**Solution:**
- For each frequency level i = 0, 1, ..., L-1:
  - sin(2^i πp) and cos(2^i πp) = 2 components
- Total: L × 2 = 10 × 2 = **20 dimensions per coordinate**
- For 3D position (x,y,z): 3 × 20 = **60 dimensions total**

---

## NeRF-based Avatars

### Extension of NeRF

**Goal:** Reconstruct and render human avatars from images

**Challenges:**
1. **Deformation:** Human bodies deform
2. **Clothing:** Complex geometry and materials
3. **Temporal consistency:** Frame-to-frame coherence

### Approaches

#### 1. Deformable NeRF
- Model deformation using skeleton or blend shapes
- NeRF conditioned on pose parameters
- **Input:** Position x, direction d, pose θ
- **Output:** Density σ, radiance c

#### 2. Neural Human Rendering
- Separate geometry and appearance
- Geometry: SDF (signed distance field) or occupancy
- Appearance: View-dependent radiance
- **Advantages:** Better geometry, re-lighting possible

#### 3. Instant Avatar Methods
- Faster training/inference
- Use hash grids or other efficient representations
- Real-time or near-real-time performance

### Key Techniques

**Pose conditioning:**
- Encode skeletal pose
- Use canonical space + deformation field

**Temporal consistency:**
- Regularization terms in loss
- Temporal smoothness constraints

**Multi-view consistency:**
- Ensure geometry consistent across views
- Use silhouette constraints

### Practice Problem 4

**Question:** A NeRF-based avatar system processes 100 input images at 1920×1080 resolution. If each image requires evaluating the neural network at 1000 sample points per pixel, how many network evaluations are needed?

**Solution:**
- Pixels per image: 1920 × 1080 = 2,073,600
- Samples per pixel: 1000
- Network evaluations per image: 2,073,600 × 1000 = 2,073,600,000
- Total for 100 images: 100 × 2,073,600,000 = **207,360,000,000 evaluations**

This demonstrates why efficient network architectures and sampling strategies are crucial!

---

## Summary: Part 4 Key Takeaways

1. **3D geometry processing** involves acquisition, processing, and application
2. **Point cloud reconstruction** converts discrete points to continuous surfaces
3. **Scene reconstruction** uses inverse rendering to recover 3D from images
4. **NeRF** represents scenes as neural radiance fields for novel view synthesis
5. **NeRF-based avatars** extend NeRF to handle deformable human subjects

---

**Next:** [Part 5: Data-Driven Shape Modeling](review_part5_shape_modeling.md) | [Previous: Part 3](review_part3_neural_rendering.md) | [Back to Index](REVIEW_INDEX.md)

