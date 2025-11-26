# Part 1: Rendering Fundamentals

## Table of Contents
1. [Introduction to Realistic Rendering](#introduction)
2. [Acceleration Structures](#acceleration-structures)
3. [Radiometry](#radiometry)
4. [Monte Carlo Integration](#monte-carlo-integration)

---

## Introduction to Realistic Rendering

### Key Concepts

**Goal of Realistic Rendering:**
- Simulate the physical behavior of light to generate images that appear photorealistic
- Model light transport from light sources through the scene to the camera

**Rendering Strategies:**
1. **Rendering Pipeline (Rasterization)**: Object-order algorithm, Z-buffering
   - Fast, suitable for real-time (games, VR)
   - Limited global illumination
2. **Ray Tracing**: Image-order algorithm
   - Full global illumination possible
   - Most popular for photo-realistic rendering
   - Recently with hardware support

**Ray Tracing Pseudocode:**
```
for each pixel:
    ray = computePrimaryViewRay(pixel)
    hit = first intersection of ray with scene
    color = shade(hit)  // using shadow ray
    set pixel color
```

**Computational Complexity:**
- Without acceleration: O(n) per ray, where n = number of primitives
- Total cost: objects × rays
- Example: 1024×1024 image, 1000 triangles = 10⁹ ray-triangle intersections

**Camera Setup:**
- **Extrinsic matrix**: Transforms from world to camera coordinates [u v w e]
- **Intrinsic matrix**: Maps pixel (i,j) to ray in camera coordinates
- **Viewing frustum**: Defined by vertical field-of-view θ, aspect ratio
- **Primary rays**: Through pixel centers or random positions within pixels

**Ray-Surface Intersection:**
- **Implicit surfaces**: f(x,y,z) = 0, solve f(p(t)) = 0
- **Parametric surfaces**: p = (x(u,v), y(u,v), z(u,v)), solve ray = surface
- **Triangles**: Use Cramer's rule, check barycentric coordinates

**Surface Normals:**
- **Implicit**: n = ∇f / ||∇f||
- **Parametric**: n = (∂p/∂u × ∂p/∂v) / ||∂p/∂u × ∂p/∂v||

### Important Terms

- **Global Illumination**: Accounting for all light interactions (direct + indirect)
- **Path Tracing**: Technique for sampling light paths
- **Ray Tracing**: Geometric technique for finding intersections
- **Soft shadows, caustics, indirect illumination**: Effects requiring global illumination

---

## Acceleration Structures

### Purpose
Accelerate ray-scene intersection queries, which are the computational bottleneck in ray tracing.

### Key Data Structures

#### 1. Bounding Volume Hierarchy (BVH)

**Concept:**
- Hierarchical tree structure
- Each node contains a bounding volume (AABB - Axis-Aligned Bounding Box)
- Scene objects are partitioned into smaller groups

**Construction:**
- Recursively partition space
- Split along longest axis
- Stop when node contains few objects (leaf node)

**Query:**
- Traverse tree from root
- Test ray against bounding volumes
- Only test objects in intersected nodes

**Time Complexity:**
- Construction: O(n log n)
- Query: O(log n) average case

**Sketch:**
```
        [Root AABB]
       /           \
  [Left AABB]   [Right AABB]
   /      \       /      \
[Obj1] [Obj2] [Obj3] [Obj4]
```

#### 2. Spatial Subdivision

**Uniform Grid:**
- Partition space into uniform cells
- Each cell stores references to overlapping objects
- Traverse grid along ray
- **Advantages**: Simple, can stop at first hit
- **Disadvantages**: "Teapot in stadium" problem, may intersect same object multiple times

**Hierarchical Grid / Octree:**
- Recursively subdivide space
- Octree: Split cubic cell into 8 sub-cells
- **Advantages**: Adaptive to scene density
- **Disadvantages**: More complex implementation

**k-d Tree:**
- Binary space partitioning tree
- Alternates splitting along x, y, z axes
- Creates axis-aligned planes
- **Properties**: More memory efficient than BVH, good for static geometry

#### 3. Spatial Data Structures Comparison

| Structure | Construction | Query | Memory | Dynamic Updates |
|-----------|--------------|-------|--------|-----------------|
| BVH | O(n log n) | O(log n) | Medium | Moderate |
| k-d Tree | O(n log² n) | O(log n) | Low | Difficult |
| Grid | O(n) | O(1) avg | High | Easy |

### Practice Problem 1

**Question:** Given a scene with 1,000,000 triangles, how many intersection tests would be needed without acceleration structures? With a BVH (assuming log₂(1,000,000) ≈ 20 levels)?

**Solution:**
- Without acceleration: Up to 1,000,000 tests per ray
- With BVH: Approximately 20 × (average objects per leaf) tests
- If leaf nodes contain ~10 objects: ~200 tests per ray
- **Speedup: ~5,000x** (theoretical, actual depends on tree quality)

---

## Radiometry

### Fundamental Quantities

#### 1. Radiant Energy (Q)
- Total energy emitted, transmitted, or received
- Units: Joules (J)

#### 2. Radiant Power / Flux (Φ)
- Energy per unit time: Φ = dQ/dt
- Units: Watts (W)

#### 3. Irradiance (E)
- Power per unit area **arriving** at a surface
- E = dΦ/dA
- Units: W/m²

**Key Point:** Irradiance decreases with distance squared (inverse square law)

#### 4. Radiance (L)
- Power per unit area per unit solid angle
- L = d²Φ / (dA dω cos θ)
- Units: W/(m²·sr)

**Why Radiance is Fundamental:**
- Radiance is **conserved** along a ray in vacuum
- This makes it the natural quantity for light transport

### Solid Angle

**Definition:**
- 2D angle in 3D space
- Analogous to angles (radians) in 2D
- ω = A/r² (for small angles)
- Units: Steradians (sr)

**Full sphere:** 4π steradians

**Integration over Sphere:**
- Given function f(ω) over sphere
- ∫_sphere f(ω) dω
- Parameterize using spherical coordinates (θ, φ)
- dω = sin θ dθ dφ
- ∫_sphere f(ω) dω = ∫_0^π ∫_0^(2π) f(θ,φ) sin θ dθ dφ

**Example:** ∫_sphere 1 dω = 4π (surface area of unit sphere)

### Bidirectional Reflectance Distribution Function (BRDF)

**Definition:**
```
f_r(ω_i → ω_o) = dL_o(ω_o) / (L_i(ω_i) cos θ_i dω_i)
```

**Physical Properties:**
1. **Reciprocity**: f_r(ω_i → ω_o) = f_r(ω_o → ω_i)
2. **Energy Conservation**: ∫_hemisphere f_r(ω_i → ω_o) cos θ_i dω_i ≤ 1

### Rendering Equation (Preview)

The rendering equation relates outgoing radiance to incoming radiance:

```
L_o(x, ω_o) = L_e(x, ω_o) + ∫_hemisphere f_r(x, ω_i → ω_o) L_i(x, ω_i) cos θ_i dω_i
```

Where:
- L_o: Outgoing radiance
- L_e: Emitted radiance
- L_i: Incoming radiance
- f_r: BRDF
- θ_i: Angle between ω_i and surface normal

### Practice Problem 2

**Question:** A point light source emits 100W uniformly in all directions. What is the irradiance at a point 2 meters away on a surface perpendicular to the light direction?

**Solution:**
- Power: Φ = 100W
- Distance: r = 2m
- Surface area of sphere at distance r: A = 4πr² = 4π(2)² = 16π m²
- Irradiance: E = Φ/A = 100W / (16π m²) ≈ **1.99 W/m²**

**Follow-up:** If the surface is tilted 45° from perpendicular, what is the irradiance?
- E_tilted = E × cos(45°) = 1.99 × 0.707 ≈ **1.41 W/m²**

---

## Monte Carlo Integration

### Motivation

Many rendering integrals cannot be solved analytically:
- Complex BRDFs
- Complex geometry
- Multiple bounces

Monte Carlo provides a **numerical solution**.

### Basic Monte Carlo Estimator

For an integral: I = ∫_a^b f(x) dx

**Monte Carlo estimate (uniform sampling):**
```
I ≈ (b-a)/N × Σ_{i=1}^N f(X_i)
```

Where X_i ~ Uniform[a,b] are random samples.

**General form (with PDF p(x)):**
```
I = ∫ f(x) dx = ∫ [f(x)/p(x)] p(x) dx ≈ (1/N) Σ_{i=1}^N f(X_i)/p(X_i)
```

Where X_i ~ p(x) (samples from distribution with PDF p).

**Key Properties:**
- **Unbiased**: E[Î] = I
- **Convergence**: Error decreases as O(1/√N)
- **Variance**: Var[Î] = (1/N) Var[f(X)/p(X)]

### Key Properties

1. **Unbiased**: E[Î] = I (expected value equals true integral)
2. **Convergence**: Error decreases as O(1/√N)
3. **Variance**: Var[Î] = (1/N) Var[f(X)/p(X)]

### Importance Sampling

**Goal:** Reduce variance by sampling more where f(x) is large.

**Optimal PDF:** p*(x) = f(x)/I (proportional to integrand)

**Variance reduction:**
- Uniform sampling: Var ∝ ∫ f²(x) dx
- Importance sampling: Var ∝ ∫ f²(x)/p(x) dx

### Example: Integrating a Function

**Problem:** Estimate I = ∫_0^1 x² dx = 1/3

**Uniform Sampling:**
- Sample X_i ~ Uniform[0,1]
- Estimate: Î = (1/N) Σ X_i²
- Variance: Var = (1/N) × (1/5 - 1/9) = 4/(45N)

**Importance Sampling (p(x) = 2x):**
- Sample X_i from p(x) = 2x (using inverse CDF: X = √U)
- Estimate: Î = (1/N) Σ X_i²/(2X_i) = (1/N) Σ X_i/2
- Lower variance!

### Practice Problem 3

**Question:** Estimate ∫_0^π sin(x) dx = 2 using Monte Carlo with:
a) Uniform sampling on [0,π]
b) Importance sampling with p(x) = (2/π) sin(x)

**Solution:**

**a) Uniform Sampling:**
- Sample X_i ~ Uniform[0,π]
- Estimate: Î = (π/N) Σ sin(X_i)
- Expected value: E[Î] = π × (2/π) = 2 ✓

**b) Importance Sampling:**
- Sample from p(x) = (2/π) sin(x)
- CDF: P(x) = (2/π)∫_0^x sin(t) dt = (2/π)(1 - cos(x))
- Inverse: x = arccos(1 - (π/2)u) where u ~ Uniform[0,1]
- Estimate: Î = (1/N) Σ sin(X_i) / [(2/π) sin(X_i)] = (1/N) Σ π/2 = π/2
- **Wait, this is wrong!** Let me recalculate...

Actually, for importance sampling:
- Î = (1/N) Σ f(X_i)/p(X_i) = (1/N) Σ sin(X_i) / [(2/π) sin(X_i)] = (1/N) Σ π/2 = π/2

But the integral should be 2, not π/2. The issue is that p(x) needs to be normalized. Let me fix:

**Corrected Importance Sampling:**
- p(x) = (1/2) sin(x) (normalized: ∫_0^π p(x) dx = 1)
- CDF: P(x) = (1/2)(1 - cos(x))
- Inverse: x = arccos(1 - 2u)
- Estimate: Î = (1/N) Σ sin(X_i) / [(1/2) sin(X_i)] = (1/N) Σ 2 = 2 ✓

### Sampling PDFs: The Inversion Method

**Problem:** Generate samples X from arbitrary PDF p(x)

**1D Continuous Case:**
1. Compute CDF: P(x) = ∫_{-∞}^x p(t) dt
2. Compute inverse: P⁻¹(u) where u ~ Uniform[0,1]
3. Sample: X = P⁻¹(U) where U ~ Uniform[0,1]

**Discrete Case:**
1. Compute discrete CDF: P_i = Σ_{j=0}^i p_j
2. Sample u ~ Uniform[0,1]
3. Find i such that P_{i-1} < u ≤ P_i

### 2D Sampling (Sample Warping)

**Goal:** Generate samples on desired 2D domain (disk, hemisphere, etc.) with desired density

**Approach:**
1. Start with canonical uniform random variables ξ₁, ξ₂
2. Transform to desired domain and density

**Example: Cosine-weighted hemisphere sampling**
- Uniform samples on disk → map to hemisphere
- Results in p(ω) = cos(θ)/π distribution

**Theory:**
- Express PDF in convenient coordinates
- Transform PDF to sampling coordinates using Jacobian
- Use marginal and conditional PDFs: p(x,y) = p(x) p(y|x)
- Sample each 1D PDF using inversion method

### Variance Reduction Techniques

1. **Importance Sampling**: Sample proportional to integrand
2. **Stratified Sampling**: Divide domain into strata, sample uniformly in each
   - **Jittered sampling**: Random sample in each grid cell
   - **Latin hypercube (N-rooks)**: One sample per row/column
3. **Quasi-Monte Carlo**: Use low-discrepancy sequences (Halton, Sobol)
   - Better convergence: O((log N)^d / N) vs O(1/√N)
   - Deterministic (not random)
4. **Multiple Importance Sampling (MIS)**: Combine multiple sampling strategies (covered in Part 2)

### Practice Problem 4

**Question:** Compare the variance of estimating ∫_0^1 e^x dx using:
a) Uniform sampling
b) Importance sampling with p(x) = e^x / (e - 1)

**Solution:**

**True value:** I = e - 1 ≈ 1.718

**a) Uniform Sampling:**
- Var[Î] = (1/N) Var[e^X] where X ~ Uniform[0,1]
- Var[e^X] = E[e^(2X)] - (E[e^X])²
- E[e^X] = e - 1
- E[e^(2X)] = (1/2)(e² - 1)
- Var[e^X] = (1/2)(e² - 1) - (e - 1)² ≈ 0.242

**b) Importance Sampling:**
- Var[Î] = (1/N) Var[e^X / p(X)]
- e^X / p(X) = e^X × (e - 1) / e^X = e - 1 (constant!)
- Var = 0 (zero variance estimator!)

This demonstrates the power of optimal importance sampling.

---

## Summary: Part 1 Key Takeaways

1. **Acceleration structures** are essential for efficient ray tracing
2. **Radiance** is the fundamental quantity for light transport
3. **Monte Carlo integration** enables solving complex rendering integrals
4. **Importance sampling** is crucial for variance reduction
5. Understanding these fundamentals is essential for advanced topics

---

**Next:** [Part 2: Advanced Rendering Techniques](review_part2_advanced_rendering.md) | [Back to Index](REVIEW_INDEX.md)

