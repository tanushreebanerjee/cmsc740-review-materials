# CMSC 740 Equation Reference Sheet

Quick reference guide to all important equations organized by topic.

---

## Radiometry

### Fundamental Quantities

**Radiance:**
```
L(x,ω) = d²Φ / (dA dω cos θ)
```
- Units: W/(m²·sr)
- **Key property:** Radiance is conserved along a ray in vacuum

**Irradiance:**
```
E(x) = dΦ/dA = ∫_hemisphere L_i(x,ω_i) cos θ_i dω_i
```
- Units: W/m²
- Power per unit area arriving at surface

**Radiant Intensity:**
```
I(ω) = dΦ/dω
```
- Units: W/sr
- Power per unit solid angle

**Radiant Power/Flux:**
```
Φ = dQ/dt
```
- Units: W (Watts)
- Energy per unit time

### Solid Angle

**Definition:**
```
ω = A/r²  (for small angles)
```

**Full sphere:** 4π steradians

**Integration over sphere:**
```
∫_sphere f(ω) dω = ∫_0^π ∫_0^(2π) f(θ,φ) sin θ dθ dφ
```

**Integration over hemisphere:**
```
∫_hemisphere f(ω) dω = ∫_0^(π/2) ∫_0^(2π) f(θ,φ) sin θ dθ dφ
```

**Example - Uniform radiance:**
```
∫_hemisphere cos θ dω = ∫_0^(π/2) ∫_0^(2π) cos θ sin θ dθ dφ = π
```

---

## BRDF and Reflection

### BRDF Definition

```
f_r(x, ω_i → ω_o) = dL_o(x, ω_o) / (L_i(x, ω_i) cos θ_i dω_i)
```

### BRDF Properties

**Reciprocity (Helmholtz):**
```
f_r(ω_i → ω_o) = f_r(ω_o → ω_i)
```

**Energy Conservation:**
```
∫_hemisphere f_r(ω_i → ω_o) cos θ_i dω_i ≤ 1
```

**Positivity:**
```
f_r(ω_i → ω_o) ≥ 0
```

### Reflection Integral

```
L_o(x, ω_o) = ∫_hemisphere f_r(x, ω_i → ω_o) L_i(x, ω_i) cos θ_i dω_i
```

### Common BRDF Models

**Lambertian (Diffuse):**
```
f_r = ρ_d / π
```

**Reflection equation for Lambertian:**
```
L_o = (ρ_d/π) × ∫_hemisphere L_i cos θ_i dω_i = ρ_d × (average L_i)
```

**Perfect Specular Reflection:**
```
f_r(ω_i → ω_o) = δ(ω_o - reflect(ω_i, n))
```

**Torrance-Sparrow (Microfacet):**
```
f_r = k_d f_lambert + k_s [D(ω_h) F(ω_i,ω_o) G(ω_i,ω_o)] / (4 cos θ_i cos θ_o)
```

---

## Rendering Equation

### Hemispherical Form

```
L_o(x, ω_o) = L_e(x, ω_o) + ∫_hemisphere f_r(x, ω_i → ω_o) L_i(x, ω_i) cos θ_i dω_i
```

Where:
- L_o: Outgoing radiance (unknown)
- L_e: Emitted radiance (known, from light sources)
- L_i: Incoming radiance (unknown, comes from other surfaces)

### Operator Form

```
L = L_e + T L
```

Where T is the transport operator.

### Solution (Neumann Series)

```
L = (I - T)^(-1) L_e = Σ_{k=0}^∞ T^k L_e
```

**Physical interpretation:**
- k=0: Direct illumination (L_e)
- k=1: One bounce (T L_e)
- k=2: Two bounces (T² L_e)
- ... (infinite bounces)

### Three-Point Form

```
L_o(x → y) = L_e(x → y) + ∫_surface f_r(x, y' → y) L_o(y' → x) G(y' ↔ x) dA(y')
```

**Geometry term:**
```
G(y' ↔ x) = V(y' ↔ x) (cos θ_x cos θ_y') / ||x - y'||²
```

Where:
- V(y' ↔ x): Visibility function (1 if visible, 0 if occluded)
- θ_x, θ_y': Angles between normals and direction x→y'

**Change of variables:**
```
dω_i = (cos θ_y' / ||x - y'||²) dA(y')
```

---

## Monte Carlo Integration

### Basic Estimator

**Uniform sampling:**
```
I = ∫_a^b f(x) dx ≈ (b-a)/N × Σ_{i=1}^N f(X_i)
```
Where X_i ~ Uniform[a,b]

**General form (with PDF p(x)):**
```
I = ∫ f(x) dx ≈ (1/N) Σ_{i=1}^N f(X_i) / p(X_i)
```
Where X_i ~ p(x)

### Properties

**Unbiased:**
```
E[Î] = I
```

**Variance:**
```
Var[Î] = (1/N) Var[f(X)/p(X)]
```

**Standard deviation (error):**
```
σ = √(Var[Î]) = (1/√N) √(Var[f(X)/p(X)])
```

**Convergence:** O(1/√N)

### Optimal Importance Sampling

If p*(x) = f(x)/I (proportional to integrand):
- Var[Î] = 0 (zero variance estimator!)
- But requires knowing I (the unknown integral)

---

## Sampling PDFs

### Inversion Method (1D)

**Steps:**
1. Compute CDF: P(x) = ∫_{-∞}^x p(t) dt
2. Compute inverse: P⁻¹(u)
3. Sample: X = P⁻¹(U) where U ~ Uniform[0,1]

**Example - p(x) = 2x on [0,1]:**
- CDF: P(x) = x²
- Inverse: P⁻¹(u) = √u
- Sample: X = √U

### 2D Sampling (Sample Warping)

**Marginal and conditional:**
```
p(x,y) = p(x) p(y|x)
```

**Transform PDFs:**
```
p_Y(y) = p_X(x) / |det J_T(x)|
```
Where y = T(x) and J_T is the Jacobian.

**Example - Uniform disk:**
- Desired: p(x,y) = 1/π for x²+y² ≤ 1
- Transform to polar: (r,θ) where x = r cos θ, y = r sin θ
- Jacobian: |J| = r
- PDF in polar: p(r,θ) = r/π
- Marginal: p(r) = 2r, Conditional: p(θ|r) = 1/(2π)
- Sample: r = √u₁, θ = 2πu₂

### Cosine-Weighted Hemisphere Sampling

**Desired PDF:**
```
p(ω) = cos(θ) / π
```

**Sampling:**
- Uniform on disk: (x,y) = (√u₁ cos(2πu₂), √u₁ sin(2πu₂))
- Map to hemisphere: z = √(1 - x² - y²)
- Direction: ω = (x, y, z)

---

## Multiple Importance Sampling (MIS)

### MIS Estimator

```
Î = (1/N) Σ_{i=1}^M Σ_{j=1}^{N_i} w_i(X_{i,j}) f(X_{i,j}) / p_i(X_{i,j})
```

Where:
- M: number of strategies
- N_i: samples from strategy i
- p_i: PDF of strategy i
- w_i: Weight for strategy i

### Balance Heuristic

```
w_i(x) = p_i(x) / Σ_{j=1}^M p_j(x)
```

### Power Heuristic

```
w_i(x) = [p_i(x)]^β / Σ_{j=1}^M [p_j(x)]^β
```

Typically β = 2.

**Requirement:** Partition of unity: Σ_i w_i(x) = 1

---

## Participating Media

### Volume Rendering Equation

**Integro-integral form:**
```
L(x,ω) = L_0(x₀,ω) T(x₀→x) + ∫_0^d L_s(x',ω) T(x'→x) dt
```

**Source term:**
```
L_s(x,ω) = L_e(x,ω) + σ_s(x) ∫_sphere p(ω'→ω) L_i(x,ω') dω'
```

### Transmittance

```
T(x→y) = exp(-∫_x^y σ_t(s) ds)
```

**Homogeneous media (Beer's law):**
```
T = exp(-σ_t d)
```

Where:
- σ_t = σ_a + σ_s: Extinction coefficient
- σ_a: Absorption coefficient
- σ_s: Scattering coefficient

### Optical Thickness

```
τ(x→y) = ∫_x^y σ_t(s) ds
```

**Transmittance:**
```
T = exp(-τ)
```

### Phase Function

**Properties:**
- Normalized: ∫_sphere p(ω'→ω) dω = 1
- Reciprocity: p(ω'→ω) = p(ω→ω')

**Isotropic:**
```
p(ω'→ω) = 1/(4π)
```

**Henyey-Greenstein:**
```
p(cos θ) = (1 - g²) / [4π (1 + g² - 2g cos θ)^(3/2)]
```
Where g ∈ [-1,1] is anisotropy parameter.

---

## Path Tracing

### Path Contribution (Three-Point Form)

For path x₀ → x₁ → ... → xₖ:

```
C = L_e(y₀→y₁) × [Π_{i=1}^{k-1} G and f_r terms] × W_e
```

### Russian Roulette

**Termination probability:** q[k] at step k

**Unbiased estimator:**
- If terminated: contribution = 0
- If not terminated: weight by 1/(1-q[k])

**Expected value unchanged:**
```
E[contribution] = (1-q) × (contribution/(1-q)) = contribution
```

---

## Bidirectional Path Tracing (BDPT)

### Path Notation

**Path of length k:**
- k+1 vertices: x₀, x₁, ..., xₖ
- s vertices from light, t from eye, s+t = k+1

### Connection Strategies

For path length k, there are k+2 strategies:
- s = 0,1,...,k+1 (connect after s light vertices)

### MIS Weights

**Balance heuristic:**
```
w_{s,t} = p_{s,t} / Σ_{i,j} p_{i,j}
```

Where p_{s,t} is probability to sample path with strategy (s,t).

---

## NeRF (Neural Radiance Fields)

### Volumetric Rendering

**Color of ray r(t) = o + td:**
```
C(r) = ∫_0^d T(t) σ(r(t)) c(r(t), d) dt
```

**Transmittance:**
```
T(t) = exp(-∫_0^t σ(r(s)) ds)
```

### Numerical Approximation

**Ray marching:**
```
C(r) ≈ Σ_{i=1}^N T_i (1 - exp(-σ_i δ)) c_i
```

Where:
- T_i = exp(-Σ_{j=0}^{i-1} σ_j δ): Probability to reach segment i
- (1 - exp(-σ_i δ)): Probability of emission in segment i
- δ = d/N: Step size

### Positional Encoding

**For coordinate p:**
```
γ(p) = (sin(2⁰πp), cos(2⁰πp), ..., sin(2^(L-1)πp), cos(2^(L-1)πp))
```

**Dimensions:**
- Per coordinate: 2L dimensions
- For 3D position: 6L dimensions
- Typically: L=10 for position, L=4 for direction

---

## Neural Radiosity

### Traditional Radiosity

**Radiosity equation (diffuse only):**
```
B(x) = E(x) + ρ(x) ∫_M B(y) G(x,y) dA(y)
```

**Discretized (linear system):**
```
B = E + F·B
```

Where:
- B: Radiosity vector (per mesh face)
- E: Emitted radiosity
- F: Form factor matrix

**Solution:**
```
B = (I - F)^(-1) E
```

### Neural Radiosity

**Continuous representation:**
- Neural network: x → B(x)
- No mesh discretization needed
- Can handle general BRDFs

---

## Subsurface Scattering

### BSSRDF

**Definition:**
```
S(x_i, ω_i, x_o, ω_o) = dL_o(x_o, ω_o) / dΦ_i(x_i, ω_i)
```

**Difference from BRDF:**
- BRDF: x_i = x_o (same point)
- BSSRDF: x_i ≠ x_o (different points)

### Dipole Model

**Approximation for subsurface scattering:**
- Model material as semi-infinite half-space
- Place virtual light sources (dipole) below surface
- Compute radiance using diffusion approximation

**Key parameters:**
- σ_a: Absorption coefficient
- σ_s: Scattering coefficient
- σ_t = σ_a + σ_s: Extinction coefficient
- Albedo: α = σ_s / σ_t

---

## Quick Reference: Key Constants and Conversions

**Solid angle:**
- Full sphere: 4π steradians
- Hemisphere: 2π steradians

**Integration results:**
- ∫_hemisphere cos θ dω = π
- ∫_hemisphere dω = 2π
- ∫_sphere dω = 4π

**Monte Carlo convergence:**
- Error: O(1/√N)
- To halve error: need 4× samples

**Complexity:**
- BVH construction: O(n log n)
- BVH query: O(log n)
- Path tracing: O(samples × log n) per pixel

---

## Common Integrals

**∫_0^π sin(x) dx = 2**

**∫_0^π cos(x) sin(x) dx = 0**

**∫_0^(π/2) cos(θ) sin(θ) dθ = 1/2**

**∫_0^(π/2) sin(θ) dθ = 1**

**∫_0^1 x dx = 1/2**

**∫_0^1 x² dx = 1/3**

---

**Note:** This reference sheet should be used alongside the detailed review materials for complete understanding.

