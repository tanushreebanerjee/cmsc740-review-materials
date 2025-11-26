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

**Definition:**
- Dual graph of Voronoi diagram
- Delaunay vertex (input point) corresponding to each Voronoi cell
- Delaunay edge joining two neighboring Voronoi cells

**Properties:**
- No Delaunay vertex lies inside any circumsphere of any Delaunay triangle
- Maximizes minimum angle in all triangles (Delaunay criterion)
- **Advantages:** Guaranteed to exist, unique
- **Disadvantages:** May create unwanted triangles

**Algorithms:**
- Non-trivial, especially in higher dimensions
- Standard implementations available (e.g., Qhull)

#### 1b. Voronoi Diagram

**Definition:**
- Partition of space into regions given by set of points
- Voronoi cell of each input point: area closest to that point
- Voronoi vertex: equidistant to three or more input points
- Voronoi edge: equidistant to two input points

**Medial Axis/Surface:**
- Set of points with more than one closest point on curve/surface
- Extension of Voronoi diagram to continuous curves/surfaces
- For 2D curves: Voronoi vertices of dense samples approximate medial axis
- For 3D: Doesn't hold (Voronoi vertices don't necessarily lie close to medial axis)

#### 1c. Crust Algorithm

**Goal:** Guarantee watertight mesh, same topology (genus) as original shape

**Curve Reconstruction (2D):**
1. Compute Voronoi diagram of sample points
2. Compute Delaunay triangulation of sample points and Voronoi vertices
3. Crust: all edges of Delaunay triangulation that contain only sample points (Voronoi filtering)

**3D Extension:**
- Use Voronoi vertices farthest from sample (instead of all Voronoi vertices)
- More complex, see Amenta et al. paper

#### 2. Poisson Surface Reconstruction

**Key Paper:** "Screened Poisson Surface Reconstruction" (Kazhdan & Hoppe, ACM TOG 2013)

**Approach:**
- Fit implicit function f to point cloud
- Solve Poisson equation: ∇²f = ∇·V
- Extract isosurface {x | f(x) = 0} using marching cubes

**Marching Cubes Algorithm:**
- Sample implicit function on 3D grid
- Construct triangulation of each grid cell separately
- Given values of f at grid vertices
- Triangulate zero-isosurface
- Visit all cells, triangulate each cell with sign changes at vertices
- 2⁸ = 256 cases of possible sign configurations
- Can reduce to 15 basic cases using symmetries (rotation, reflection)
- Encode each cell as 8-bit number (signs at vertices)
- Look-up triangulation in precomputed table
- Position vertices along cube edges by linear interpolation

**Advantages:** Robust to noise, produces watertight surfaces
**Disadvantages:** May fill holes incorrectly

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
5. **Static scenes only**
6. **Training and rendering slow**
7. **Requires many input images**

### NeRF-W (NeRF in the Wild)

**Key Paper:** "NeRF in the Wild: Neural Radiance Fields for Unconstrained Photo Collections" (CVPR 2021)

**Goal:** NeRF using images from different cameras, times, conditions

**Challenges:**
1. Different times of day, atmospheric conditions, imaging pipelines
2. Transient objects (people, cars, etc.)

**Solutions:**

#### 1. Latent Appearance Model
- Per-image latent appearance vector l_i
- Represents time of day, atmospheric conditions, camera settings
- Image-dependent radiance: c_i(x, d, l_i)

#### 2. Transient Objects
- Separate static and transient radiance fields
- Static: c(x, d), σ(x)
- Transient: c_i^(τ)(x, d), σ_i^(τ)(x)
- Uncertainty β_i(x) to weight loss

**Loss function:**
```
L_i(r) = w_i(r) ||C_i(r) - Ĉ_i(r)||² + λ_u (1/w_i(r)) + λ_τ σ_i^(τ)(r)
```

Where w_i(r) = 1/β_i(r) (uncertainty weighting)

### MipNeRF

**Key Paper:** "Mip-NeRF: A Multiscale Representation for Anti-Aliasing Neural Radiance Fields"

**Key Idea:**
- Radiance should take into account pixel size (anti-aliasing)
- Extend spatial encoding to capture local size of conical beam
- Pixel modeled as conical beam instead of infinitesimal ray
- Generalized spatial encoding g(m, S) where:
  - m: center of ellipsoid
  - S: size of ellipsoid to locally approximate beam

**Advantages:**
- Better anti-aliasing
- More accurate reconstruction

### RefNeRF

**Key Paper:** "Ref-NeRF: Structured View-Dependent Appearance for Neural Radiance Fields" (CVPR 2022)

**Goal:** Small step towards modeling light reflection in NeRF framework

**Intuition:**
- Reflected light is (roughly) sum of diffuse and glossy
- Diffuse is view independent
- Glossy mostly determined by incident light around mirror reflection of viewing direction

**Approach:**
- Represent radiance as sum of diffuse c_d and glossy c_s
- For glossy: use directional MLP using mirror reflection of view direction as input
- **Integrated Directional Encoding (IDE):** Clever encoding of mirror reflection direction, including estimate of surface roughness
- Tone mapping function to map radiance to captured pixel colors
- Predict surface normals with regularization terms

### Nerfies (Deformable NeRF)

**Key Paper:** "Nerfies: Deformable Neural Radiance Fields" (2021)

**Goal:** Handle dynamic/deformable scenes

**Approach:**
- Include temporal space deformation
- NeRF in canonical space + deformation field
- Map sample points to canonical space using deformation model
- Similar idea to Neural Actor but more general

### Block-NeRF

**Extension to large scenes:**
- Train multiple NeRFs separately (one per city block)
- Combine for rendering
- Uses Mip-NeRF (beam geometry) + NeRF-W (appearance codes)
- Visibility prediction to select relevant NeRFs
- Appearance matching between NeRFs using latent appearance vectors

### NeuS: Neural Implicit Surfaces

**Key Paper:** "NeuS: Learning Neural Implicit Surfaces by Volume Rendering for Multi-view Reconstruction"

**Goal:** Reconstruct well-defined surfaces (not just volumetric density)

**Approach:**
- Optimize neural SDF f(p) instead of density σ
- Still use volumetric rendering similar to NeRF
- Challenge: Relationship between SDF and density for volumetric rendering

**Components:**
- Neural SDF: f(p) → signed distance value
- Neural radiance field: c(p, v) → color
- Rendering function with weight function w(t)

**Weight function w(t):**
- **Unbiased:** w(t) has local maximum when f(p(t)) = 0
- **Occlusion aware:** For t₀ < t₁ where f(t₀) = f(t₁), then w(t₀) > w(t₁)
- Color contribution strongest from points on surface

**Training:**
- Include Eikonal term to regularize SDF: ||∇f|| = 1
- Hierarchical sampling similar to NeRF
- Better surface reconstruction than NeRF

### Instant Neural Graphics Primitives (Instant NGP)

**Key Paper:** "Instant Neural Graphics Primitives with a Multiresolution Hash Encoding" (2022)

**Goal:** Faster training and rendering than original NeRF

**Approach: Hybrid grid/neural network**
- Store learnable feature vectors on multi-resolution grid (using spatial hashing)
- Use small neural network that takes feature vectors as input
- Produces radiance and density

**Architecture:**
- Multi-resolution hash grid storing feature vectors
- Spatial hashing for sparse grid storage
- Linear interpolation and concatenation of feature vectors over multiple resolutions
- Small MLP to predict radiance and density

**Advantages:**
- Much faster training than original NeRF
- Higher quality than spatial encoding
- No collision handling needed for spatial hashing (still works)

### Plenoxels: Radiance Fields without Neural Networks

**Key Paper:** "Plenoxels: Radiance Fields without Neural Networks" (CVPR 2022)

**Approach:**
- Grid-based representation instead of neural network
- Parameters: radiance, density values at grid points c_i, σ_i
- Evaluation: c(x) = linear interpolation of {c_i | i in neighborhood of x}

**Directional Dependence: Spherical Harmonics**
- Spherical harmonics: generalization of Fourier basis functions to sphere
- "Frequency" l, "phase" m, spherical angles θ, φ
- Represent functions f over sphere using weighted sum of spherical harmonics
- Weights (coefficients) f_{ml} are parameters
- Store vector of coefficients per grid point
- Typically k ≤ 3 (≤ 16 coefficients)

**Challenges:**
- Dense grid requires a lot of memory
- Solution: Sparse grids using spatial hashing
- Allocate storage only where needed (non-zero values)

### Gaussian Splatting (3D Gaussian Splatting)

**Key Paper:** "3D Gaussian Splatting for Real-Time Radiance Field Rendering" (2023)

**Approach:**
- Represent radiance field using 3D Gaussians instead of neural network
- Parameters: coefficients c_i, covariances Σ_i, centers x_i
- Radial basis functions: weighted sum of Gaussians

**Volumetric rendering:**
- Same volume rendering model as NeRF
- Color and density: c(r(t)), σ(r(t)) from Gaussian evaluation
- Ray marching through Gaussians
- Alpha-blend projected, integrated Gaussians based on densities

**Advantages:**
- Very fast rendering (real-time)
- High quality
- Efficient representation
- Main benefit: fast rendering

**Training:**
- Optimize Gaussian parameters (positions, covariances, colors)
- Adaptive density control (add/remove Gaussians)
- Efficient ray marching requires spatial data structures

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

### SMPL (Skinned Multi-Person Linear Model)

**Key Paper:** "SMPL: A Skinned Multi-Person Linear Model" (ACM TOG 2015)

**Purpose:** Parametric model for pose- and body shape-dependent geometry

**Function:**
```
geometry = SMPL(body pose, body shape parameters)
```

**Based on:** Skeletal animation (rigging and skinning)

#### Skeletal Animation Basics

**Rigging:**
- Template mesh in reference pose
- Skeleton: joints connected via bones (kinematic chain)
- Skin weights: influence of each bone on each vertex (matrix W)

**Linear Blend Skinning (LBS):**
```
x'_i = Σ_b w_{bi} T_b x_i
```

Where:
- x_i: Vertex i in reference pose
- x'_i: Deformed vertex
- w_{bi}: Weight of bone b on vertex i
- T_b: Transformation matrix of bone b

**Forward Kinematics:**
- Compute T_b from joint angles
- Step-by-step along kinematic chain from root
- Better to use dual quaternions instead of matrices

#### SMPL Extensions

**Limitations of basic LBS:**
- Cannot match detailed deformed shapes
- Single template, no identity-specific shape

**SMPL solution:**
1. **Identity-specific blend shapes:**
   ```
   B_S(β) = Σ_n β_n S_n
   ```
   - S_n: Pre-trained basis vectors
   - β_n: Shape coefficients (fit to data)
   - Typically 10 shape coefficients

2. **Pose-dependent blend shapes:**
   ```
   B_P(θ) = Σ_n R_n(θ) P_n
   ```
   - P_n: Pre-trained pose-dependent vectors
   - R(θ): Function mapping 23 joint angles to rotation matrices
   - Uses Rodrigues' formula

3. **Final geometry:**
   ```
   vertices = T + B_S(β) + B_P(θ)
   deformed = LBS(vertices, W, θ)
   ```

**SMPL Parameters:**
- N = 6890 vertices
- K = 23 joints
- Template mesh T, blend weights W (pre-trained)
- Identity blend shapes S_n (pre-trained)
- Pose-dependent blend shapes P_n (pre-trained)
- Shape coefficients β (fit per person)
- Joint angles θ (fit per pose)

### Neural Actor Approach

**Goal:** Learn rendering function for virtual human from videos

**Architecture:**
```
image = f(camera, geometry, appearance)
geometry = SMPL(pose, shape)
```

**Key idea:** Inverse deformation to canonical space
- NeRF in canonical (fixed pose) space
- Map sample points to canonical space by inverting LBS
- Use skinning weights from closest surface point

**Training:**
- Multi-view video (11-12 cameras, ~30,000 frames)
- Infer pose from images
- Optimize appearance parameters

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

