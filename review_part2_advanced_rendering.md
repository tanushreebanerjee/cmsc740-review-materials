# Part 2: Advanced Rendering Techniques

## Table of Contents
1. [BRDF and Reflection Integral](#brdf-and-reflection-integral)
2. [Rendering Equation](#rendering-equation)
3. [Advanced Sampling Techniques](#advanced-sampling-techniques)
4. [Bidirectional Path Tracing (BDPT)](#bidirectional-path-tracing)
5. [Participating Media and Subsurface Scattering](#participating-media)

---

## BRDF and Reflection Integral

### BRDF Definition

**Bidirectional Reflectance Distribution Function:**
```
f_r(x, ω_i → ω_o) = dL_o(x, ω_o) / (L_i(x, ω_i) cos θ_i dω_i)
```

**Physical Interpretation:**
- Ratio of outgoing radiance to incoming irradiance
- Describes how light reflects at a surface point

### Key BRDF Properties

#### 1. Reciprocity (Helmholtz Reciprocity)
```
f_r(ω_i → ω_o) = f_r(ω_o → ω_i)
```
- Light path is reversible
- Based on physics (time-reversal symmetry)

#### 2. Energy Conservation
```
∫_hemisphere f_r(ω_i → ω_o) cos θ_i dω_i ≤ 1
```
- Cannot reflect more energy than received
- Equality holds for perfect reflectors

#### 3. Positivity
```
f_r(ω_i → ω_o) ≥ 0
```
- BRDF values are always non-negative

### Reflection Integral

**Outgoing radiance from reflection:**
```
L_o(x, ω_o) = ∫_hemisphere f_r(x, ω_i → ω_o) L_i(x, ω_i) cos θ_i dω_i
```

**Coordinate System:**
- Use local coordinate frame aligned with surface normal
- Spherical coordinates: (θ, φ) where θ is angle from normal

### Common BRDF Models

#### 1. Lambertian (Diffuse)
```
f_r = ρ_d / π
```
- Constant BRDF (independent of direction)
- ρ_d: diffuse albedo (reflectance), fraction of reflected over incoming light
- Perfectly matte surface
- Reflection equation: L_o = ρ_d/π × ∫_hemisphere L_i cos θ_i dω_i

#### 2. Perfect Specular Reflection (Mirror)
- BRDF involves Dirac delta function
- Reflection: ω_o = reflect(ω_i, n)
- f_r(ω_i → ω_o) = δ(ω_o - reflect(ω_i, n))

#### 3. Perfect Specular Refraction
- Snell's law: n₁ sin θ₁ = n₂ sin θ₂
- BRDF involves Dirac delta function
- Total internal reflection when critical angle exceeded

#### 4. Torrance-Sparrow (Microfacet Model)
```
f_r = k_d f_lambert + k_s D(ω_h) F(ω_i, ω_o) G(ω_i, ω_o) / (4 cos θ_i cos θ_o)
```
- D: Normal Distribution Function (microfacet orientation)
- F: Fresnel term (reflectance at interface)
- G: Geometry term (shadowing/masking)
- ω_h: Half-vector between ω_i and ω_o

**Blinn Microfacet Distribution:**
- D(ω_h) ∝ (cos θ_h)^e where e is shininess coefficient
- Used for importance sampling specular highlights

#### 5. Phong Model
- Not physically plausible, may violate energy conservation
- f_r = k_d (ρ_d/π) + k_s (n+2)/(2π) (cos α)^n

### Practice Problem 1

**Question:** For a Lambertian surface with albedo ρ_d = 0.8, illuminated by uniform radiance L_i = 1.0 from the hemisphere above, what is the outgoing radiance in the normal direction?

**Solution:**
- BRDF: f_r = ρ_d / π = 0.8 / π
- Reflection integral: L_o = ∫ f_r L_i cos θ_i dω_i
- For uniform L_i: L_o = f_r L_i ∫ cos θ_i dω_i
- ∫_hemisphere cos θ_i dω_i = ∫_0^(π/2) ∫_0^(2π) cos θ sin θ dθ dφ = π
- L_o = (0.8/π) × 1.0 × π = **0.8**

**Key insight:** For Lambertian, outgoing radiance = albedo × average incoming radiance

---

## Rendering Equation

### Full Rendering Equation

```
L_o(x, ω_o) = L_e(x, ω_o) + ∫_hemisphere f_r(x, ω_i → ω_o) L_i(x, ω_i) cos θ_i dω_i
```

**Components:**
- **L_e**: Emitted radiance (light sources)
- **L_i**: Incoming radiance (from other surfaces)
- **f_r**: BRDF (surface reflection)
- **cos θ_i**: Cosine term (projected area)

### Recursive Nature

The rendering equation is **recursive**:
- L_i at point x comes from L_o at other points
- This creates infinite recursion (light bounces)

**Operator Form:**
```
L = L_e + T L
```
Where T is the transport operator.

**Solution:**
```
L = (I - T)^(-1) L_e = Σ_{k=0}^∞ T^k L_e
```

**Physical Interpretation:**
- k=0: Direct illumination (L_e)
- k=1: One bounce
- k=2: Two bounces
- ... (infinite bounces)

### Solving the Rendering Equation

#### 1. Path Tracing (Forward)
- Start from camera
- Trace paths backward
- Sample light sources and BRDF

**Path Tracing Pseudocode:**
```
// Main rendering loop
for each pixel:
    color = 0
    for i = 1 to N:  // N samples per pixel
        alpha = 1
        hitRecord = shootPrimaryRay(pixel)
        k = 0
        while true:
            // Next event estimation: connect to light
            lightPoint = sampleLight()
            color += alpha * shade(hitRecord, lightPoint) / pdf(lightPoint)
            
            // Russian roulette termination
            if random() < q[k]:
                break
            
            // Sample next direction (BRDF sampling)
            direction = sampleBRDF(hitRecord)
            hitRecord = shootRay(hitRecord, direction)
            alpha = alpha * BRDF * cos / (pdf(direction) * (1 - q[k]))
            k++
    color = color / N
    setPixel(pixel, color)
```

**Key components:**
- `alpha`: Accumulated throughput (product of BRDFs and cosines)
- `q[k]`: Russian roulette termination probability at depth k
- `shade(hitRecord, lightPoint)`: Computes contribution from light point
- Typically: q[0] = q[1] = 0 (never terminate early), q[k] = 0.5 for k > 1

#### 2. Light Tracing (Backward)
- Start from light sources
- Trace paths forward
- Connect to camera

#### 3. Bidirectional Path Tracing
- Build paths from both camera and light
- Connect paths in the middle

### Three-Point Form

**Alternative formulation using geometry:**
```
L_o(x → y) = L_e(x → y) + ∫_surface f_r(x, y' → y) L_o(y' → x) G(y' ↔ x) dA(y')
```

**Geometry Term:**
```
G(y' ↔ x) = V(y' ↔ x) cos θ_x cos θ_y' / ||x - y'||²
```
- V: Visibility function (1 if visible, 0 if occluded)
- Accounts for distance and angles

**Advantages:**
- Explicitly handles occlusion
- More suitable for some algorithms (e.g., photon mapping)

### Practice Problem 2

**Question:** Derive the three-point form from the standard rendering equation.

**Solution:**

Start with standard form:
```
L_o(x, ω_o) = L_e(x, ω_o) + ∫_hemisphere f_r(x, ω_i → ω_o) L_i(x, ω_i) cos θ_i dω_i
```

Change of variables: dω_i = (cos θ_y' / ||x - y'||²) dA(y')
- Solid angle to area element
- y' is the point in direction ω_i from x

Substitute:
```
L_o(x, ω_o) = L_e(x, ω_o) + ∫_surface f_r(x, y' → y) L_i(x, y') (cos θ_x cos θ_y' / ||x - y'||²) dA(y')
```

L_i(x, y') comes from L_o(y' → x), and we need visibility:
```
L_i(x, y') = V(y' ↔ x) L_o(y' → x)
```

Therefore:
```
L_o(x → y) = L_e(x → y) + ∫_surface f_r(x, y' → y) L_o(y' → x) V(y' ↔ x) (cos θ_x cos θ_y' / ||x - y'||²) dA(y')
```

Which is the three-point form with G(y' ↔ x) = V(y' ↔ x) (cos θ_x cos θ_y' / ||x - y'||²).

---

## Advanced Sampling Techniques

### Surface Form of Reflection Equation

**Change of variables:** From solid angle to surface area integration

**Geometry term:**
```
G(x ↔ y) = V(x ↔ y) (cos θ_x cos θ_y) / ||x - y||²
```

Where:
- V(x ↔ y): Visibility function (1 if visible, 0 if occluded)
- θ_x, θ_y: Angles between surface normals and direction x→y

**Surface form:**
```
L_o(x, ω_o) = L_e(x, ω_o) + ∫_surface f_r(x, y → x) L_o(y → x) G(x ↔ y) dA(y)
```

**Advantages:**
- Explicitly handles occlusion
- More suitable for some algorithms (photon mapping, BDPT)

### Multiple Importance Sampling (MIS)

**Problem:** Different sampling strategies work well in different scenarios:
- **BSDF sampling**: Good for glossy surfaces
- **Light sampling**: Good when light is small
- How to combine them?

**Naive approach:** Average both (F_BSDF + F_light)/2
- Variance bounded by larger variance of two techniques

**MIS Solution:** Weighted average with optimal weights

**MIS Estimator:**
```
Î = (1/N) Σ_{i=1}^M Σ_{j=1}^{N_i} w_i(X_{i,j}) f(X_{i,j}) / p_i(X_{i,j})
```

Where:
- M: number of strategies
- N_i: samples from strategy i
- p_i: PDF of strategy i
- w_i: Weight for strategy i

**Requirement:** Partition of unity: Σ_i w_i(x) = 1

#### Balance Heuristic

**Weight:**
```
w_i(x) = p_i(x) / Σ_j p_j(x)
```

**Properties:**
- Unbiased estimator
- Provable variance reduction
- Simple to implement

#### Power Heuristic

**Weight:**
```
w_i(x) = [p_i(x)]^β / Σ_j [p_j(x)]^β
```

Typically β = 2.

**Advantage:** More aggressive weighting, further variance reduction

**Implementation:**
- Need to compute probability densities for each sample under all strategies
- Convert between surface area and solid angle densities: p_solid_angle = p_area / (cos × r²)

**Example: Direct Illumination**
- Take two samples for each next event estimation:
  - One by sampling BRDF/BSDF (next randomly sampled ray direction)
  - One by sampling the light source
- Combine using MIS weights

**MIS Implementation Pseudocode:**
```
function computeMISContribution(sample, strategies):
    // strategies = list of (pdf_function, sample_value) pairs
    total_pdf = 0
    for (pdf_func, _) in strategies:
        total_pdf += pdf_func(sample)
    
    contributions = []
    for (pdf_func, sample_val) in strategies:
        weight = pdf_func(sample) / total_pdf  // Balance heuristic
        contribution = weight * f(sample_val) / pdf_func(sample_val)
        contributions.append(contribution)
    
    return sum(contributions) / len(strategies)
```

### Beyond Uniform Random Numbers

**Problem:** Uniform random samples can have large gaps and dense clusters, leading to high variance.

**Solutions:**

#### 1. Stratified Sampling
- **Idea:** Divide sample space into strata, place one sample per stratum
- **Jittered sampling:** Uniform grid, random sample in each cell
- **Advantages:** Reduces clumping, lower variance
- **Disadvantages:** Not suitable for high dimensions (curse of dimensionality)

**Stratified Sampling Pseudocode:**
```
function stratifiedSample(n, dim):
    samples = []
    grid_size = ceil(n^(1/dim))
    for each cell in grid:
        offset = randomUniform()  // jitter within cell
        sample = cell_center + offset * cell_size
        samples.append(sample)
    return samples
```

#### 2. N-Rooks (Latin Hypercube) Sampling
- **Idea:** Ensure each dimension has exactly one sample per "row"
- **Advantage:** Can generate any number n of samples (not restricted to n×m grid)
- **Better for higher dimensions than jittered grid**

#### 3. Quasi-Monte Carlo (QMC)
- **Idea:** Use low-discrepancy sequences instead of pseudo-random numbers
- **Low-discrepancy sequences:** More uniform distribution, less clumping
- **Examples:** Halton sequence, Hammersley sequence
- **Advantage:** Theoretically better convergence than Monte Carlo
- **Can substitute directly for pseudo-random points in [0,1]^n**

**Key insight:** All improved sampling sequences yield points (ξ₁, ξ₂, ..., ξₙ) in unit hypercube [0,1]^n, can directly substitute for pseudo-random points.

### Practice Problem 3

**Question:** Estimate the reflection integral for a surface where:
- BRDF is moderately glossy
- Light source is small but bright
- We take 1 sample from BRDF (p₁) and 1 sample from light (p₂)
- f/p₁ = 10, f/p₂ = 2

Compute MIS estimate using balance heuristic.

**Solution:**

**Sample 1 (BRDF):**
- p₁(X₁) = 0.1, p₂(X₁) = 0.01
- w₁(X₁) = 0.1 / (0.1 + 0.01) = 0.909
- Contribution: w₁(X₁) × (f/p₁) = 0.909 × 10 = 9.09

**Sample 2 (Light):**
- p₁(X₂) = 0.05, p₂(X₂) = 0.2
- w₂(X₂) = 0.2 / (0.05 + 0.2) = 0.8
- Contribution: w₂(X₂) × (f/p₂) = 0.8 × 2 = 1.6

**MIS Estimate:** Î = (1/2) × (9.09 + 1.6) = **5.345**

**Comparison:**
- BRDF only: Î = 10 (high variance)
- Light only: Î = 2 (high variance)
- MIS: Î = 5.345 (lower variance by combining both)

### Beyond Uniform Random Sampling

#### 1. Quasi-Monte Carlo (QMC)
- Use low-discrepancy sequences (Halton, Sobol)
- Better convergence: O((log N)^d / N) vs O(1/√N)
- Deterministic (not random)

#### 2. Stratified Sampling
- Divide domain into strata
- Sample uniformly in each stratum
- Reduces variance by ensuring coverage

#### 3. Metropolis Sampling
- Markov Chain Monte Carlo (MCMC)
- Good for complex, high-dimensional distributions
- Used in some advanced rendering algorithms

---

## Three-Point Form and Measurement Equation

### Three-Point Form of Rendering Equation

**Surface area integration instead of solid angle:**

**Change of variables:**
```
dω_i = (cos θ_y' / ||x - y'||²) dA(y')
```

**Three-point form:**
```
L_o(x → y) = L_e(x → y) + ∫_surface f_r(x, y' → y) L_o(y' → x) G(y' ↔ x) dA(y')
```

Where:
- **Geometry term:** G(y' ↔ x) = V(y' ↔ x) (cos θ_x cos θ_y' / ||x - y'||²)
- **Visibility function:** V(y' ↔ x) = 1 if visible, 0 if occluded

**Advantages:**
- Explicitly handles occlusion
- More flexible for path sampling strategies
- Enables bidirectional path tracing

### Measurement Equation

**Pixel value I_j of pixel j:**
```
I_j = ∫_hemisphere W_j(x, ω) L(x, ω) cos θ dω dA
```

Where:
- **W_j:** Importance function for pixel j (box function: 1 inside pixel, 0 elsewhere)
- **L(x, ω):** Radiance

**Surface area form:**
```
I_j = ∫_surface ∫_surface W_j(x → y) L_o(x → y) G(x ↔ y) dA(x) dA(y)
```

### Recursive Expansion

**Neumann series expansion:**
```
L_o = L_e + T L_e + T² L_e + T³ L_e + ...
```

**Path contribution function:**
- Path of length k: x₀ (light) → x₁ → ... → xₖ (camera)
- Contribution: product of emission, BRDFs, geometry terms, visibility

**Monte Carlo estimation:**
- Sample random paths
- Path probabilities: product of vertex probabilities p(x_i) and path length probability p(k)
- In practice: construct paths via ray tracing, not direct surface sampling

### Path Tracing Revisited (Three-Point Form)

**Expressed using three-point form:**
- Sample paths incrementally from eye
- At each step, connect to light to obtain path of length k
- Terminate using Russian roulette
- Many terms in geometry/density division cancel out

**Key insight:** Three-point form is equivalent to hemispherical form, but more flexible for advanced path sampling.

## Bidirectional Path Tracing (BDPT)

### Motivation

**Limitations of unidirectional path tracing:**
- Camera path: Good for small lights (hard to hit)
- Light path: Good for small camera (hard to hit)
- **BDPT combines both!**

### Algorithm Overview

1. **Build eye subpath:** x₀ (camera) → x₁ → ... → xₖ
2. **Build light subpath:** y₀ (light) → y₁ → ... → yₗ
3. **Connect subpaths:** Connect xₖ to yₗ
4. **Weight contributions:** Use MIS to weight different connection strategies

**Path notation:**
- Path of length k (k+1 vertices): x₀ → x₁ → ... → xₖ
- Sampled with s vertices from light, t from eye: s + t = k + 1
- Path denoted: x̄_{s,t}

**Connection strategies:**
- Each length k can be sampled in k+2 ways (different s,t combinations)
- s = 0,1,...,k+1 and t = k+1-s
- Probability density for technique (s,t): p_{s,t}

**BDPT Pseudocode (Conceptual):**
```
for each pixel:
    // Build eye subpath
    eyePath = []
    current = camera
    while random() > q_eye:
        eyePath.append(current)
        current = sampleBRDF(current)
    
    // Build light subpath
    lightPath = []
    current = sampleLight()
    while random() > q_light:
        lightPath.append(current)
        current = sampleBRDF(current)
    
    // Evaluate all connections
    for s = 0 to len(eyePath):
        for t = 0 to len(lightPath):
            if canConnect(eyePath[s], lightPath[t]):
                contribution = computePathContribution(eyePath, lightPath, s, t)
                weight = computeMISWeight(eyePath, lightPath, s, t)
                pixel += weight * contribution
```

**Key points:**
- Each path length k can be sampled in k+2 ways (different connection strategies)
- MIS weights combine all strategies that could generate the same path
- More efficient for caustics and small lights

### Path Contribution

**Full path:** x₀ → x₁ → ... → xₖ → yₗ → ... → y₁ → y₀

**Contribution:**
```
C = L_e(y₀ → y₁) × [Π G and f_r terms] × W_e
```

Where W_e is the pixel filter weight.

### Connection Strategies

For a path with k eye vertices and l light vertices:

**Strategy s:** Connect eye vertex s to light vertex (k+l-s)

**All strategies:**
- s = 0: Direct connection (camera to light)
- s = 1: Connect after 1 eye bounce
- ...
- s = k: Connect after k eye bounces (standard path tracing)
- s = k+1: Connect after 1 light bounce
- ...

### MIS Weighting in BDPT

Each connection strategy is a sampling technique. Use MIS to combine:
```
Î = Σ_s w_s C_s
```

Where w_s uses balance or power heuristic over all strategies.

### Advantages

1. **Handles difficult cases:** Small lights, caustics, complex illumination
2. **Unbiased:** All strategies are valid
3. **Efficient:** Reuses subpaths for multiple connections

### Practice Problem 4

**Question:** For a scene with a small light and a mirror, explain why BDPT performs better than unidirectional path tracing.

**Solution:**

**Unidirectional Path Tracing (camera start):**
- Hard to hit small light directly
- Mirror creates caustics (focused light paths)
- Need many samples to capture these effects

**BDPT:**
- Light subpath can start from light and bounce off mirror
- Eye subpath starts from camera
- Connect them in the middle
- **Much more efficient** at capturing caustics and small light contributions

**Example:** Light → Mirror → Diffuse Surface → Camera
- Unidirectional: Very unlikely to sample this path
- BDPT: Light subpath: Light → Mirror, Eye subpath: Camera → Diffuse, Connect: Mirror ↔ Diffuse
- **Much higher probability** of finding this path

---

## Participating Media and Subsurface Scattering

### Participating Media

**Definition:** Volumes that absorb, emit, and scatter light (e.g., fog, smoke, clouds).

### Volume Rendering Equation

```
L(x, ω) = L_0(x₀, ω) T(x₀ → x) + ∫_0^d L_s(x', ω) T(x' → x) dt
```

**Components:**
- **L_0**: Radiance entering volume
- **T**: Transmittance (fraction of light that survives)
- **L_s**: Source term (emission + in-scattering)

### Transmittance

**Definition:**
```
T(x → y) = exp(-∫_x^y σ_t(s) ds)
```

Where σ_t is the **extinction coefficient** (absorption + out-scattering).

**Physical meaning:** Probability that light travels from x to y without being absorbed or scattered.

### Source Term

```
L_s(x, ω) = L_e(x, ω) + σ_s(x) ∫_sphere p(ω' → ω) L_i(x, ω') dω'
```

- **L_e**: Emission
- **σ_s**: Scattering coefficient
- **p**: Phase function (scattering distribution)

### Phase Function

Describes angular distribution of scattered light.

**Isotropic:** p(ω' → ω) = 1/(4π) (uniform)

**Henyey-Greenstein:**
```
p(cos θ) = (1 - g²) / [4π (1 + g² - 2g cos θ)^(3/2)]
```
- g ∈ [-1, 1]: anisotropy parameter
- g > 0: forward scattering
- g < 0: backward scattering

### Subsurface Scattering

**Definition:** Light enters a material, scatters internally, then exits at a different point.

**BSSRDF (Bidirectional Scattering-Surface Reflectance Distribution Function):**
```
S(x_i, ω_i, x_o, ω_o) = dL_o(x_o, ω_o) / (dΦ_i(x_i, ω_i))
```

**Difference from BRDF:**
- BRDF: Same point (x_i = x_o)
- BSSRDF: Different points (x_i ≠ x_o)

### Dipole Model

**Approximation for subsurface scattering:**
- Model material as semi-infinite half-space
- Place virtual light sources (dipole) below surface
- Compute radiance using diffusion approximation

**Key parameters:**
- **σ_a**: Absorption coefficient
- **σ_s**: Scattering coefficient
- **σ_t = σ_a + σ_s**: Extinction coefficient
- **Albedo**: α = σ_s / σ_t

### Practice Problem 5

**Question:** Light travels through fog with extinction coefficient σ_t = 0.1 m⁻¹. What fraction of light survives after traveling 10 meters?

**Solution:**

**Transmittance:**
```
T = exp(-σ_t × d) = exp(-0.1 × 10) = exp(-1) ≈ 0.368
```

**Answer:** About **36.8%** of light survives.

**Follow-up:** What distance results in 50% transmittance?
- 0.5 = exp(-0.1 × d)
- d = -ln(0.5) / 0.1 = 0.693 / 0.1 = **6.93 meters**

---

## Summary: Part 2 Key Takeaways

1. **BRDF** models surface reflection; must satisfy reciprocity and energy conservation
2. **Rendering equation** is recursive and describes global illumination
3. **MIS** combines multiple sampling strategies to reduce variance
4. **BDPT** builds paths from both camera and light for better efficiency
5. **Participating media** requires volume rendering equation with transmittance
6. **Subsurface scattering** uses BSSRDF to model light transport inside materials

---

**Next:** [Part 3: Neural Rendering](review_part3_neural_rendering.md) | [Previous: Part 1](review_part1_fundamentals.md) | [Back to Index](REVIEW_INDEX.md)

