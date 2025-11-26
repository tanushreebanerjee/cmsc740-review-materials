# CMSC 740 Practice Problems Collection

Comprehensive collection of practice problems covering all topics with detailed solutions.

**Note:** This file contains additional practice problems beyond those in the Question Bank. The Question Bank focuses on exam-style questions, while this collection includes more computational exercises.

---

## Part 1: Rendering Fundamentals

### Problem Set 1.1: Ray Tracing and Acceleration Structures

#### Problem 1.1.1
**Problem:** A scene contains 500,000 triangles. Without acceleration structures, how many intersection tests are needed for a 1920×1080 image? With a BVH of depth 19 and average 8 objects per leaf?

**Solution:**
- **Without acceleration:**
  - Tests per ray: up to 500,000
  - Total rays: 1920 × 1080 = 2,073,600
  - Total tests: 500,000 × 2,073,600 = **1.037 × 10¹² tests**

- **With BVH:**
  - Tests per ray: ~19 × 8 = 152 (depth × objects per leaf)
  - Total tests: 152 × 2,073,600 = **3.15 × 10⁸ tests**
  - **Speedup: 3,290×**

#### Problem 1.1.2
**Problem:** Compare memory usage for a BVH vs. uniform grid for a scene with 1,000,000 triangles. Assume BVH has 2,000,000 nodes (each 32 bytes) and uniform grid is 100×100×100 (each cell stores 4-byte pointer to object list, average 10 objects per cell).

**Solution:**
- **BVH:**
  - Nodes: 2,000,000 × 32 bytes = **64 MB**

- **Uniform Grid:**
  - Cells: 100³ = 1,000,000 cells
  - Pointers: 1,000,000 × 4 bytes = 4 MB
  - Object lists: 1,000,000 × 10 × 4 bytes = 40 MB
  - Total: **44 MB**

- **BVH uses more memory** but provides better query performance for most scenes.

---

### Problem Set 1.2: Radiometry

#### Problem 1.2.1
**Problem:** A point light source emits 200W uniformly. Calculate:
a) Radiant intensity
b) Irradiance at 3m on perpendicular surface
c) Irradiance at 3m on surface tilted 60° from perpendicular

**Solution:**
- **a) Radiant Intensity:**
  - For isotropic source: I = Φ/(4π) = 200W/(4π) = **15.92 W/sr**

- **b) Irradiance (perpendicular):**
  - Distance: r = 3m
  - Surface area: A = 4πr² = 4π(3)² = 36π m²
  - Irradiance: E = Φ/A = 200W/(36π m²) = **1.77 W/m²**

- **c) Irradiance (tilted 60°):**
  - E_tilted = E × cos(60°) = 1.77 × 0.5 = **0.885 W/m²**

#### Problem 1.2.2
**Problem:** A surface receives uniform radiance L_i = 2.0 W/(m²·sr) from the hemisphere above. Calculate the irradiance.

**Solution:**
- **Irradiance:**
  ```
  E = ∫_hemisphere L_i cos θ_i dω_i
    = L_i ∫_hemisphere cos θ_i dω_i
    = L_i × π
    = 2.0 × π
    = 6.28 W/m²
  ```

#### Problem 1.2.3
**Problem:** Show that radiance is conserved along a ray in vacuum. Consider two surfaces dA₁ and dA₂ connected by a ray of length r.

**Solution:**
- **Power from dA₁ to dA₂:**
  ```
  dΦ₁₂ = L₁ dA₁ cos θ₁ dω₁
  ```
  Where dω₁ = dA₂ cos θ₂ / r²

- **Therefore:**
  ```
  dΦ₁₂ = L₁ dA₁ cos θ₁ (dA₂ cos θ₂ / r²)
       = L₁ (dA₁ cos θ₁ dA₂ cos θ₂) / r²
  ```

- **By reciprocity and energy conservation:**
  ```
  dΦ₂₁ = L₂ dA₂ cos θ₂ (dA₁ cos θ₁ / r²)
       = L₂ (dA₁ cos θ₁ dA₂ cos θ₂) / r²
  ```

- **Energy conservation:** dΦ₁₂ = dΦ₂₁
- **Therefore:** L₁ = L₂ ✓

---

### Problem Set 1.3: Monte Carlo Integration

#### Problem 1.3.1
**Problem:** Estimate I = ∫_0^1 e^x dx = e - 1 ≈ 1.718 using Monte Carlo with:
a) Uniform sampling
b) Importance sampling with p(x) = e^x / (e - 1)

Compare variances.

**Solution:**
- **a) Uniform Sampling:**
  - Sample X_i ~ Uniform[0,1]
  - Estimator: Î = (1/N) Σ e^(X_i)
  - E[Î] = e - 1 ✓
  - Var[Î] = (1/N) Var[e^X] where X ~ Uniform[0,1]
  - Var[e^X] = E[e^(2X)] - (E[e^X])²
  - E[e^X] = e - 1
  - E[e^(2X)] = (1/2)(e² - 1)
  - Var[e^X] = (1/2)(e² - 1) - (e - 1)² ≈ 0.242
  - Var[Î] ≈ 0.242/N

- **b) Importance Sampling:**
  - Sample from p(x) = e^x / (e - 1)
  - CDF: P(x) = (e^x - 1)/(e - 1)
  - Inverse: x = ln(1 + u(e - 1)) where u ~ Uniform[0,1]
  - Estimator: Î = (1/N) Σ e^X / p(X) = (1/N) Σ (e - 1) = e - 1
  - **Zero variance!** (optimal importance sampling)

#### Problem 1.3.2
**Problem:** Sample from PDF p(x) = 3x² on [0,1] using inversion method.

**Solution:**
1. **CDF:**
   ```
   P(x) = ∫_0^x 3t² dt = x³
   ```

2. **Inverse:**
   ```
   P⁻¹(u) = u^(1/3)
   ```

3. **Sample:**
   ```
   X = U^(1/3) where U ~ Uniform[0,1]
   ```

**Verification:**
- PDF of X = U^(1/3):
  - P(X ≤ x) = P(U^(1/3) ≤ x) = P(U ≤ x³) = x³
  - p_X(x) = d/dx (x³) = 3x² ✓

#### Problem 1.3.3
**Problem:** Uniformly sample a hemisphere. Show that the PDF in spherical coordinates is p(θ,φ) = sin(θ)/(2π).

**Solution:**
- **Desired PDF:** p(ω) = 1/(2π) (uniform on hemisphere)
- **Transform to spherical:** (θ,φ) where θ ∈ [0,π/2], φ ∈ [0,2π]
- **Jacobian:** |J| = sin θ
- **PDF in spherical:**
  ```
  p(θ,φ) = p(ω) × sin θ = (1/(2π)) × sin θ = sin(θ)/(2π)
  ```

- **Marginal:**
  ```
  p(θ) = ∫_0^(2π) sin(θ)/(2π) dφ = sin(θ)
  ```

- **Conditional:**
  ```
  p(φ|θ) = (sin(θ)/(2π)) / sin(θ) = 1/(2π)
  ```

- **Sampling:**
  - θ: P(θ) = 1 - cos(θ), so θ = arccos(1 - u₁)
  - φ: φ = 2πu₂

---

## Part 2: Advanced Rendering

### Problem Set 2.1: BRDF and Reflection

#### Problem 2.1.1
**Problem:** A Lambertian surface with albedo ρ_d = 0.6 is illuminated by:
- Uniform radiance L_i = 1.0 from hemisphere above
- Additional point light: L_point = 5.0 in direction (0, 0.707, 0.707) (45° from normal)

Calculate total outgoing radiance in normal direction.

**Solution:**
- **From uniform illumination:**
  ```
  L_o_uniform = (ρ_d/π) × L_i × π = ρ_d × L_i = 0.6 × 1.0 = 0.6
  ```

- **From point light:**
  - Direction: ω_i = (0, 0.707, 0.707)
  - cos θ_i = 0.707
  - L_o_point = (ρ_d/π) × L_point × cos θ_i = (0.6/π) × 5.0 × 0.707 = 0.675

- **Total:**
  ```
  L_o = 0.6 + 0.675 = 1.275
  ```

#### Problem 2.1.2
**Problem:** Verify energy conservation for Lambertian BRDF: f_r = ρ_d / π. Show that ∫_hemisphere f_r cos θ_i dω_i = ρ_d.

**Solution:**
```
∫_hemisphere f_r cos θ_i dω_i = ∫_hemisphere (ρ_d/π) cos θ_i dω_i
                                = (ρ_d/π) ∫_hemisphere cos θ_i dω_i
                                = (ρ_d/π) × π
                                = ρ_d ✓
```

Since ρ_d ≤ 1, energy is conserved.

---

### Problem Set 2.2: Rendering Equation

#### Problem 2.2.1
**Problem:** Expand the Neumann series L = Σ T^k L_e for k = 0, 1, 2. Interpret each term physically.

**Solution:**
- **k = 0:** L₀ = L_e
  - Direct emission from light sources

- **k = 1:** L₁ = T L_e
  - One bounce: light emitted, then reflected once

- **k = 2:** L₂ = T² L_e
  - Two bounces: light emitted, reflected twice

- **Full solution:** L = L₀ + L₁ + L₂ + ... (infinite bounces)

---

### Problem Set 2.3: Multiple Importance Sampling

#### Problem 2.3.1
**Problem:** For direct illumination, we have two strategies:
- Strategy 1 (BSDF): p₁(x) = 0.2, f/p₁ = 8
- Strategy 2 (Light): p₂(x) = 0.3, f/p₂ = 4

Compute MIS estimate using balance heuristic.

**Solution:**
- **Balance heuristic weights:**
  - w₁ = 0.2 / (0.2 + 0.3) = 0.4
  - w₂ = 0.3 / (0.2 + 0.3) = 0.6

- **Contributions:**
  - Strategy 1: w₁ × (f/p₁) = 0.4 × 8 = 3.2
  - Strategy 2: w₂ × (f/p₂) = 0.6 × 4 = 2.4

- **MIS estimate (with 1 sample each):**
  ```
  Î = (1/2) × (3.2 + 2.4) = 2.8
  ```

---

### Problem Set 2.4: Participating Media

#### Problem 2.4.1
**Problem:** Light travels through fog with σ_t = 0.05 m⁻¹. Calculate:
a) Transmittance after 20m
b) Distance for 10% transmittance
c) Optical thickness after 20m

**Solution:**
- **a) Transmittance:**
  ```
  T = exp(-0.05 × 20) = exp(-1) ≈ 0.368
  ```
  **Answer:** 36.8% survives

- **b) 10% transmittance:**
  ```
  0.1 = exp(-0.05 × d)
  -ln(0.1) = 0.05 × d
  d = 2.303 / 0.05 = 46.06 meters
  ```

- **c) Optical thickness:**
  ```
  τ = σ_t × d = 0.05 × 20 = 1.0
  ```

#### Problem 2.4.2
**Problem:** In a homogeneous medium, show that the volume rendering equation simplifies to:
```
L(x,ω) = L_0 T + L_s (1 - T) / σ_t
```
Where L_s is the source term (assumed constant).

**Solution:**
- **Volume rendering equation:**
  ```
  L(x,ω) = L_0 T(x₀→x) + ∫_0^d L_s T(x'→x) dt
  ```

- **For homogeneous medium:**
  - T(x'→x) = exp(-σ_t (d - t))
  - L_s constant

- **Evaluate integral:**
  ```
  ∫_0^d L_s exp(-σ_t (d - t)) dt
  = L_s exp(-σ_t d) ∫_0^d exp(σ_t t) dt
  = L_s exp(-σ_t d) [exp(σ_t t)/σ_t]_0^d
  = L_s exp(-σ_t d) (exp(σ_t d) - 1)/σ_t
  = L_s (1 - exp(-σ_t d))/σ_t
  = L_s (1 - T)/σ_t
  ```

- **Therefore:**
  ```
  L = L_0 T + L_s (1 - T)/σ_t
  ```

---

## Part 3: Neural Rendering

### Problem Set 3.1: Neural Representations

#### Problem 3.1.1
**Problem:** A neural network represents a 3D shape using an SDF (signed distance field). The network takes 3D position (x,y,z) as input and outputs distance d. If the network has 5 hidden layers with 256 neurons each, approximately how many parameters does it have? (Assume input layer 3, output layer 1)

**Solution:**
- **Layer 1 (input → hidden):** 3 × 256 + 256 (bias) = 1,024
- **Layers 2-5 (hidden → hidden):** 4 × (256 × 256 + 256) = 4 × 65,792 = 263,168
- **Layer 6 (hidden → output):** 256 × 1 + 1 = 257
- **Total:** 1,024 + 263,168 + 257 = **264,449 parameters**

---

## Part 4: Geometry and Reconstruction

### Problem Set 4.1: Geometry Processing

#### Problem 4.1.1
**Problem:** A point cloud has 500,000 points. If we reconstruct a mesh with average vertex degree 6 (each vertex connected to 6 edges), estimate:
a) Number of vertices (assuming 1 vertex per point)
b) Number of edges
c) Number of faces (triangles)

**Solution:**
- **a) Vertices:** 500,000

- **b) Edges:**
  - Each edge connects 2 vertices
  - Total edge-vertex connections: 500,000 × 6 = 3,000,000
  - Edges: 3,000,000 / 2 = **1,500,000 edges**

- **c) Faces:**
  - Each triangle has 3 edges
  - Each edge shared by 2 faces (in closed mesh)
  - Face-edge connections: 1,500,000 × 2 = 3,000,000
  - Faces: 3,000,000 / 3 = **1,000,000 triangles**

---

### Problem Set 4.2: NeRF

#### Problem 4.2.1
**Problem:** In NeRF, we use positional encoding with L=10 for 3D positions. How many input dimensions does the network receive for a single 3D point?

**Solution:**
- **Per coordinate:** 2L = 2 × 10 = 20 dimensions
- **For 3D position (x,y,z):** 3 × 20 = **60 dimensions**

#### Problem 4.2.2
**Problem:** A NeRF renders a 1920×1080 image. If we use 128 samples per ray, how many network evaluations are needed per image?

**Solution:**
- **Pixels:** 1920 × 1080 = 2,073,600
- **Samples per pixel:** 128
- **Network evaluations:** 2,073,600 × 128 = **265,420,800 evaluations**

This demonstrates why efficient network architectures are crucial!

---

## Part 5: Shape Modeling

### Problem Set 5.1: Generative Models

#### Problem 5.1.1
**Problem:** A VAE has:
- Encoder: 3D shape (10,000 vertices × 3 floats) → latent (128 dim)
- Decoder: latent (128 dim) → 3D shape (10,000 vertices × 3 floats)

Estimate number of parameters if encoder/decoder each have 3 hidden layers with 512 neurons.

**Solution:**
- **Encoder:**
  - Input: 30,000 (10,000 × 3)
  - Layer 1: 30,000 × 512 + 512 = 15,360,512
  - Layer 2: 512 × 512 + 512 = 262,656
  - Layer 3: 512 × 128 + 128 = 65,664
  - Total encoder: 15,688,832

- **Decoder:**
  - Layer 1: 128 × 512 + 512 = 65,664
  - Layer 2: 512 × 512 + 512 = 262,656
  - Layer 3: 512 × 30,000 + 30,000 = 15,360,000
  - Total decoder: 15,688,320

- **Total:** 15,688,832 + 15,688,320 = **31,377,152 parameters**

---

## Mixed Practice Problems

### Problem M.1
**Problem:** Derive the three-point form of the rendering equation from the hemispherical form, showing all steps.

**Solution:**
1. **Start with hemispherical form:**
   ```
   L_o(x, ω_o) = L_e(x, ω_o) + ∫_hemisphere f_r(x, ω_i → ω_o) L_i(x, ω_i) cos θ_i dω_i
   ```

2. **Change of variables:**
   - From solid angle to surface area
   - dω_i = (cos θ_y' / ||x - y'||²) dA(y')
   - L_i(x, ω_i) = V(y' ↔ x) L_o(y' → x)

3. **Substitute:**
   ```
   L_o(x, ω_o) = L_e(x, ω_o) + ∫_surface f_r(x, y' → y) V(y' ↔ x) L_o(y' → x) 
                 × (cos θ_x cos θ_y' / ||x - y'||²) dA(y')
   ```

4. **Define geometry term:**
   ```
   G(y' ↔ x) = V(y' ↔ x) (cos θ_x cos θ_y') / ||x - y'||²
   ```

5. **Three-point form:**
   ```
   L_o(x → y) = L_e(x → y) + ∫_surface f_r(x, y' → y) L_o(y' → x) G(y' ↔ x) dA(y')
   ```

### Problem M.2
**Problem:** Compare the variance of estimating ∫_0^1 x² dx = 1/3 using:
a) Uniform sampling
b) Importance sampling with p(x) = 2x

**Solution:**
- **a) Uniform sampling:**
  - Var[Î] = (1/N) Var[X²] where X ~ Uniform[0,1]
  - E[X²] = 1/3
  - E[X⁴] = 1/5
  - Var[X²] = 1/5 - (1/3)² = 1/5 - 1/9 = 4/45
  - Var[Î] = 4/(45N)

- **b) Importance sampling:**
  - Sample from p(x) = 2x
  - CDF: P(x) = x², so X = √U
  - Estimator: Î = (1/N) Σ X²/(2X) = (1/N) Σ X/2
  - E[Î] = E[X/2] = (1/2) × (2/3) = 1/3 ✓
  - Var[Î] = (1/N) Var[X/2] = (1/(4N)) Var[X]
  - E[X] = 2/3, E[X²] = 1/2
  - Var[X] = 1/2 - (2/3)² = 1/18
  - Var[Î] = 1/(72N)

- **Variance reduction:** (4/45) / (1/72) = 6.4× lower variance with importance sampling

---

**Note:** Work through these problems step-by-step. Understanding the process is more important than memorizing answers!

