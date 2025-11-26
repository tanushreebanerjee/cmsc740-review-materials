# CMSC 740 Final Exam Question Bank

**Final Exam Date:** Thursday, December 18, 10:30am - 12:30pm, IRB2107

This question bank covers all topics from the course and includes questions requiring text answers, equations, sketches, and calculations.

**Organization:** Questions are organized by major topics matching the course structure.

---

## Topic 1: Introduction & Ray Tracing Basics

#### Question 1.1
**Question:** Explain the difference between the rendering pipeline (rasterization) and ray tracing. What are the advantages and disadvantages of each approach? Include a sketch showing how each algorithm processes a scene.

**Answer:**
- **Rendering Pipeline (Rasterization):**
  - Object-order algorithm: processes one primitive (triangle) at a time
  - Transforms, projects, rasterizes each triangle
  - Uses Z-buffer for visibility
  - **Advantages:** Fast, suitable for real-time (games, VR), good memory access patterns
  - **Disadvantages:** Limited global illumination, difficult to handle complex light transport

- **Ray Tracing:**
  - Image-order algorithm: processes one pixel at a time
  - Shoots rays from camera through pixels
  - Finds intersections with scene
  - **Advantages:** Full global illumination possible, handles complex light transport
  - **Disadvantages:** Random memory access, requires acceleration structures, slower

**Sketch:**
```
Rasterization:          Ray Tracing:
Triangle 1 → Pixel      Pixel → Ray → Intersection
Triangle 2 → Pixel      Pixel → Ray → Intersection
Triangle 3 → Pixel      Pixel → Ray → Intersection
```

#### Question 1.2
**Question:** Write pseudocode for basic ray tracing. What is the computational complexity per ray without acceleration structures? With a BVH?

**Answer:**
```
for each pixel:
    ray = computePrimaryViewRay(pixel)
    hit = firstIntersection(ray, scene)
    color = shade(hit)
    setPixel(pixel, color)
```

**Complexity:**
- Without acceleration: O(n) per ray, where n = number of primitives
- With BVH: O(log n) average case
- Total: Without = O(n × pixels), With = O(log n × pixels)

#### Question 1.3
**Question:** Describe how to compute a primary viewing ray for pixel (i,j) in an image. Include the transformation from pixel coordinates to camera coordinates to world coordinates.

**Answer:**
1. **Pixel to image plane (u,v):**
   - Image resolution: m × n pixels
   - u = l + (r-l)(i+0.5)/m
   - v = b + (t-b)(j+0.5)/n
   - Where l,r,b,t are image plane bounds

2. **Image plane to camera ray:**
   - Ray in camera coords: p(t) = (0,0,0) + t(u, v, -1)
   - Intrinsic matrix K maps pixel to ray

3. **Camera to world:**
   - Extrinsic matrix [u v w e] transforms camera coords to world
   - u,v,w are camera basis vectors, e is camera position
   - Final ray: p_world(t) = e + t(u·u_cam + v·v_cam + w·w_cam)

---

## Topic 2: Acceleration Structures

#### Question 2.1
**Question:** Explain how a Bounding Volume Hierarchy (BVH) accelerates ray-scene intersection. Draw a diagram showing a BVH tree structure for 4 objects. What is the time complexity for construction and query?

**Answer:**
- **How it works:**
  1. Hierarchical tree structure
  2. Each node contains bounding volume (AABB)
  3. Leaf nodes contain objects
  4. If ray doesn't intersect bounding volume, skip entire subtree

- **Diagram:**
```
        [Root AABB]
       /           \
  [Left AABB]   [Right AABB]
   /      \       /      \
[Obj1] [Obj2] [Obj3] [Obj4]
```

- **Complexity:**
  - Construction: O(n log n)
  - Query: O(log n) average case

#### Question 2.2
**Question:** Compare BVH, uniform grid, and octree acceleration structures. When would you use each?

**Answer:**
- **BVH:**
  - Object subdivision
  - O(n log n) construction, O(log n) query
  - Good for general scenes, moderate memory
  - **Use when:** General purpose, dynamic scenes possible

- **Uniform Grid:**
  - Spatial subdivision, uniform cells
  - O(n) construction, O(1) average query
  - High memory, "teapot in stadium" problem
  - **Use when:** Uniform object distribution

- **Octree:**
  - Hierarchical spatial subdivision
  - Adaptive to scene density
  - More complex implementation
  - **Use when:** Highly non-uniform scene density

#### Question 2.3
**Question:** A scene contains 1,000,000 triangles. Without acceleration, how many intersection tests are needed per ray? With a BVH of depth 20, approximately how many tests are needed? Calculate the speedup.

**Answer:**
- Without acceleration: Up to 1,000,000 tests per ray
- With BVH (depth 20, ~10 objects per leaf): ~20 × 10 = 200 tests per ray
- Speedup: 1,000,000 / 200 = **5,000×**

---

## Topic 3: Radiometry

#### Question 3.1
**Question:** Define radiance, irradiance, and radiant intensity. Write their mathematical definitions and explain the physical meaning of each.

**Answer:**
- **Radiance L(x,ω):**
  - Definition: L = d²Φ / (dA dω cos θ)
  - Units: W/(m²·sr)
  - Physical meaning: Power per unit area per unit solid angle
  - **Key property:** Radiance is conserved along a ray in vacuum

- **Irradiance E(x):**
  - Definition: E = dΦ/dA = ∫_hemisphere L_i(x,ω_i) cos θ_i dω_i
  - Units: W/m²
  - Physical meaning: Power per unit area arriving at surface
  - **Note:** Cosine term accounts for projected area

- **Radiant Intensity I(ω):**
  - Definition: I = dΦ/dω
  - Units: W/sr
  - Physical meaning: Power per unit solid angle
  - For isotropic point source: I = constant, Φ = 4πI

#### Question 3.2
**Question:** A point light source emits 100W uniformly in all directions. Calculate the irradiance at a point 2 meters away on a surface perpendicular to the light direction. What if the surface is tilted 45°?

**Answer:**
- **Perpendicular surface:**
  - Total power: Φ = 100W
  - Distance: r = 2m
  - Surface area of sphere: A = 4πr² = 4π(2)² = 16π m²
  - Irradiance: E = Φ/A = 100W / (16π m²) ≈ **1.99 W/m²**

- **Tilted 45°:**
  - E_tilted = E × cos(45°) = 1.99 × 0.707 ≈ **1.41 W/m²**

#### Question 3.3
**Question:** Explain why radiance is the fundamental quantity for light transport. Show that radiance is conserved along a ray in vacuum.

**Answer:**
- **Why fundamental:**
  1. Radiance is what sensors (cameras, eyes) measure
  2. Radiance is conserved along rays (in vacuum)
  3. Makes light transport calculations tractable

- **Conservation proof:**
  - Consider two surfaces dA₁ and dA₂ connected by ray
  - Power from dA₁ to dA₂: dΦ₁₂ = L₁ dA₁ cos θ₁ dω₁
  - Solid angle: dω₁ = dA₂ cos θ₂ / r²
  - Power: dΦ₁₂ = L₁ dA₁ cos θ₁ (dA₂ cos θ₂ / r²)
  - By reciprocity: dΦ₂₁ = L₂ dA₂ cos θ₂ (dA₁ cos θ₁ / r²)
  - Energy conservation: dΦ₁₂ = dΦ₂₁
  - Therefore: **L₁ = L₂** (radiance conserved)

#### Question 3.4
**Question:** Write the integral for irradiance over a hemisphere. Convert it to spherical coordinates and evaluate for uniform radiance L_i = constant.

**Answer:**
- **Hemispherical integral:**
  ```
  E = ∫_hemisphere L_i(x,ω_i) cos θ_i dω_i
  ```

- **Spherical coordinates:**
  ```
  E = ∫_0^(π/2) ∫_0^(2π) L_i(θ,φ) cos θ sin θ dθ dφ
  ```

- **Uniform radiance:**
  ```
  E = L_i ∫_0^(π/2) ∫_0^(2π) cos θ sin θ dθ dφ
    = L_i × 2π × ∫_0^(π/2) cos θ sin θ dθ
    = L_i × 2π × [sin²θ/2]_0^(π/2)
    = L_i × 2π × (1/2)
    = π L_i
  ```

---

## Topic 4: Monte Carlo Integration

#### Question 4.1
**Question:** Derive the Monte Carlo estimator for the integral I = ∫_a^b f(x) dx. Show that it is unbiased and derive the variance.

**Answer:**
- **Estimator:**
  - Sample X_i ~ Uniform[a,b]
  - Î = (b-a)/N × Σ_{i=1}^N f(X_i)

- **Unbiased:**
  ```
  E[Î] = (b-a)/N × Σ_{i=1}^N E[f(X_i)]
       = (b-a)/N × N × (1/(b-a)) ∫_a^b f(x) dx
       = ∫_a^b f(x) dx = I
  ```

- **Variance:**
  ```
  Var[Î] = Var[(b-a)/N × Σ f(X_i)]
         = ((b-a)/N)² × N × Var[f(X)]
         = (b-a)²/(N) × Var[f(X)]
  ```

#### Question 4.2
**Question:** Estimate I = ∫_0^π sin(x) dx = 2 using Monte Carlo with:
a) Uniform sampling on [0,π]
b) Importance sampling with p(x) = (1/2) sin(x)

Show your work and compare the variances.

**Answer:**
- **a) Uniform sampling:**
  - Sample X_i ~ Uniform[0,π]
  - Î = (π/N) Σ sin(X_i)
  - E[Î] = π × (2/π) = 2 ✓
  - Var[Î] = π²/N × Var[sin(X)] where X ~ Uniform[0,π]

- **b) Importance sampling:**
  - Sample from p(x) = (1/2) sin(x) (normalized)
  - CDF: P(x) = (1/2)(1 - cos(x))
  - Inverse: x = arccos(1 - 2u) where u ~ Uniform[0,1]
  - Î = (1/N) Σ sin(X_i) / [(1/2) sin(X_i)] = (1/N) Σ 2 = 2
  - **Zero variance!** (optimal importance sampling)

#### Question 4.3
**Question:** Explain the inversion method for sampling from an arbitrary 1D PDF p(x). Work through an example: sample from p(x) = 2x on [0,1].

**Answer:**
- **Method:**
  1. Compute CDF: P(x) = ∫_0^x p(t) dt
  2. Compute inverse: P⁻¹(u)
  3. Sample: X = P⁻¹(U) where U ~ Uniform[0,1]

- **Example: p(x) = 2x on [0,1]**
  1. CDF: P(x) = ∫_0^x 2t dt = x²
  2. Inverse: P⁻¹(u) = √u
  3. Sample: X = √U where U ~ Uniform[0,1]

**Verification:**
- PDF of X = √U: p_X(x) = d/dx P(X ≤ x) = d/dx P(U ≤ x²) = d/dx (x²) = 2x ✓

#### Question 4.4
**Question:** Describe how to uniformly sample a unit disk. Show the transformation from uniform random variables to polar coordinates, then to Cartesian coordinates.

**Answer:**
- **Step 1: Desired PDF in Euclidean coords:**
  - p(x,y) = 1/π for x²+y² ≤ 1

- **Step 2: Transform to polar coordinates:**
  - Transformation: (x,y) = (r cos θ, r sin θ)
  - Jacobian: |J| = r
  - PDF in polar: p(r,θ) = p(x,y) × r = r/π

- **Step 3: Marginal and conditional:**
  - Marginal: p(r) = ∫_0^(2π) r/π dθ = 2r
  - Conditional: p(θ|r) = (r/π) / (2r) = 1/(2π)

- **Step 4: Sample using inversion:**
  - Sample r: P(r) = r², so r = √u₁ where u₁ ~ Uniform[0,1]
  - Sample θ: θ = 2πu₂ where u₂ ~ Uniform[0,1]

- **Step 5: Convert to Cartesian:**
  - x = r cos θ = √u₁ cos(2πu₂)
  - y = r sin θ = √u₁ sin(2πu₂)

#### Question 4.5
**Question:** Compare uniform random sampling, stratified sampling, and quasi-Monte Carlo. What are the convergence rates?

**Answer:**
- **Uniform Random:**
  - Convergence: O(1/√N)
  - Random samples, may have clumping
  - Simple to implement

- **Stratified Sampling:**
  - Convergence: O(1/√N) (same rate, but lower constant)
  - Divide domain into strata, one sample per stratum
  - Reduces clumping, better coverage
  - Examples: Jittered, Latin hypercube

- **Quasi-Monte Carlo:**
  - Convergence: O((log N)^d / N) where d is dimension
  - Uses low-discrepancy sequences (Halton, Sobol)
  - Deterministic (not random)
  - Better for low dimensions

---

---

## Topic 5: BRDF and Reflection Integral

#### Question 5.1
**Question:** Define the BRDF mathematically and explain its physical meaning. State the three key properties that BRDFs must satisfy.

**Answer:**
- **Definition:**
  ```
  f_r(x, ω_i → ω_o) = dL_o(x, ω_o) / (L_i(x, ω_i) cos θ_i dω_i)
  ```
  - Ratio of outgoing radiance to incoming irradiance
  - Describes how light reflects at a surface point

- **Properties:**
  1. **Reciprocity (Helmholtz):** f_r(ω_i → ω_o) = f_r(ω_o → ω_i)
  2. **Energy Conservation:** ∫_hemisphere f_r(ω_i → ω_o) cos θ_i dω_i ≤ 1
  3. **Positivity:** f_r(ω_i → ω_o) ≥ 0

#### Question 5.2
**Question:** Derive the reflection integral from the BRDF definition. Show how it integrates incident light over the hemisphere.

**Answer:**
- **From BRDF definition:**
  ```
  dL_o = f_r(ω_i → ω_o) L_i(ω_i) cos θ_i dω_i
  ```

- **Integrate over hemisphere:**
  ```
  L_o(ω_o) = ∫_hemisphere f_r(ω_i → ω_o) L_i(ω_i) cos θ_i dω_i
  ```

- **In spherical coordinates:**
  ```
  L_o(ω_o) = ∫_0^(π/2) ∫_0^(2π) f_r(θ_i,φ_i → θ_o,φ_o) L_i(θ_i,φ_i) cos θ_i sin θ_i dθ_i dφ_i
  ```

- **Physical interpretation:** Sum (integrate) contributions from all incident directions

#### Question 5.3
**Question:** For a Lambertian surface with albedo ρ_d = 0.8, calculate the outgoing radiance when illuminated by uniform radiance L_i = 1.0 from the hemisphere above.

**Answer:**
- **Lambertian BRDF:** f_r = ρ_d / π = 0.8 / π
- **Reflection integral:**
  ```
  L_o = ∫_hemisphere f_r L_i cos θ_i dω_i
      = f_r L_i ∫_hemisphere cos θ_i dω_i
      = (0.8/π) × 1.0 × π
      = 0.8
  ```
- **Result:** L_o = **0.8**

**Key insight:** For Lambertian, outgoing radiance = albedo × average incoming radiance

#### Question 5.4
**Question:** Explain the difference between diffuse, glossy, and specular reflection. Give examples of BRDF models for each.

**Answer:**
- **Diffuse:**
  - Light scattered uniformly in all directions
  - Example: Matte paint, paper
  - BRDF: f_r = ρ_d / π (Lambertian)

- **Glossy:**
  - Light scattered in a lobe around specular direction
  - Example: Plastic, glossy paint
  - BRDF: Torrance-Sparrow, Blinn-Phong

- **Specular:**
  - Perfect mirror reflection (or refraction)
  - Example: Mirror, glass
  - BRDF: Dirac delta function, f_r(ω_i → ω_o) = δ(ω_o - reflect(ω_i))

---

## Topic 6: Rendering Equation

#### Question 6.1
**Question:** Write the full rendering equation and explain each term. Show how it is recursive and why this creates a challenge.

**Answer:**
- **Rendering equation:**
  ```
  L_o(x, ω_o) = L_e(x, ω_o) + ∫_hemisphere f_r(x, ω_i → ω_o) L_i(x, ω_i) cos θ_i dω_i
  ```

- **Terms:**
  - L_o: Outgoing radiance (unknown)
  - L_e: Emitted radiance (known, from light sources)
  - f_r: BRDF (known, material property)
  - L_i: Incoming radiance (unknown, comes from other surfaces)

- **Recursion:**
  - L_i at point x comes from L_o at other points
  - Creates infinite recursion: L_o depends on L_i, which depends on L_o, etc.

- **Challenge:**
  - Cannot solve analytically for complex scenes
  - Need numerical methods (Monte Carlo, path tracing)

#### Question 6.2
**Question:** Derive the operator form of the rendering equation and show the Neumann series solution. Explain what each term in the series represents physically.

**Answer:**
- **Operator form:**
  ```
  L = L_e + T L
  ```
  Where T is the transport operator.

- **Solution:**
  ```
  L = (I - T)^(-1) L_e = Σ_{k=0}^∞ T^k L_e
  ```

- **Physical interpretation:**
  - k=0: Direct illumination (L_e)
  - k=1: One bounce (T L_e)
  - k=2: Two bounces (T² L_e)
  - ... (infinite bounces)

- **Convergence:** Guaranteed by energy conservation (||T|| < 1)

#### Question 6.3
**Question:** Write pseudocode for path tracing to solve the rendering equation. Explain Russian roulette and why it's used.

**Answer:**
```
for each pixel:
    color = 0
    for i = 1 to N:  // N samples per pixel
        alpha = 1
        hit = shootPrimaryRay(pixel)
        k = 0
        while true:
            // Connect to light (next event estimation)
            lightPoint = sampleLight()
            color += alpha * shade(hit, lightPoint) / pdf(lightPoint)
            
            // Russian roulette termination
            if random() < q[k]:
                break
            
            // Sample next direction
            direction = sampleBRDF(hit)
            hit = shootRay(hit, direction)
            alpha = alpha * BRDF * cos / (pdf(direction) * (1 - q[k]))
            k++
    color = color / N
```

**Russian Roulette:**
- Probabilistic termination to avoid infinite paths
- Probability q[k] to terminate at step k
- If not terminated, weight contribution by 1/(1-q[k])
- **Unbiased:** Expected value unchanged
- **Efficient:** Longer paths (less contribution) more likely to terminate

#### Question 6.4
**Question:** Derive the three-point form of the rendering equation from the hemispherical form. Show the geometry term and explain the change of variables.

**Answer:**
- **Hemispherical form:**
  ```
  L_o(x, ω_o) = L_e(x, ω_o) + ∫_hemisphere f_r(x, ω_i → ω_o) L_i(x, ω_i) cos θ_i dω_i
  ```

- **Change of variables:**
  - From solid angle to surface area
  - dω_i = (cos θ_y' / ||x - y'||²) dA(y')
  - L_i(x, ω_i) = V(y' ↔ x) L_o(y' → x)

- **Three-point form:**
  ```
  L_o(x → y) = L_e(x → y) + ∫_surface f_r(x, y' → y) L_o(y' → x) G(y' ↔ x) dA(y')
  ```

- **Geometry term:**
  ```
  G(y' ↔ x) = V(y' ↔ x) (cos θ_x cos θ_y') / ||x - y'||²
  ```

- **Advantages:**
  - Explicitly handles occlusion (V term)
  - More suitable for some algorithms (BDPT, photon mapping)

---

## Topic 7: Advanced Sampling Techniques (MIS)

#### Question 7.1
**Question:** Explain Multiple Importance Sampling (MIS). Write the MIS estimator and explain the balance heuristic. Why is MIS better than using a single sampling strategy?

**Answer:**
- **Problem:** Different strategies work in different scenarios
  - BSDF sampling: Good for glossy surfaces
  - Light sampling: Good for small lights
  - Neither works well alone

- **MIS Estimator:**
  ```
  Î = (1/N) Σ_{i=1}^M Σ_{j=1}^{N_i} w_i(X_{i,j}) f(X_{i,j}) / p_i(X_{i,j})
  ```

- **Balance Heuristic:**
  ```
  w_i(x) = p_i(x) / Σ_j p_j(x)
  ```

- **Why better:**
  - Combines strengths of multiple strategies
  - Provable variance reduction
  - Unbiased estimator
  - Adapts to scene characteristics

#### Question 7.2
**Question:** For a surface with moderately glossy BRDF and a small bright light, we take 1 sample from BRDF (p₁) and 1 sample from light (p₂). Given f/p₁ = 10 and f/p₂ = 2, and p₁(X₁) = 0.1, p₂(X₁) = 0.01, p₁(X₂) = 0.05, p₂(X₂) = 0.2, compute the MIS estimate using balance heuristic.

**Answer:**
- **Sample 1 (BRDF):**
  - w₁(X₁) = 0.1 / (0.1 + 0.01) = 0.909
  - Contribution: 0.909 × 10 = 9.09

- **Sample 2 (Light):**
  - w₂(X₂) = 0.2 / (0.05 + 0.2) = 0.8
  - Contribution: 0.8 × 2 = 1.6

- **MIS Estimate:**
  ```
  Î = (1/2) × (9.09 + 1.6) = 5.345
  ```

- **Comparison:**
  - BRDF only: 10 (high variance)
  - Light only: 2 (high variance)
  - MIS: 5.345 (lower variance)

---

## Topic 8: Bidirectional Path Tracing (BDPT)

#### Question 8.1
**Question:** Explain how Bidirectional Path Tracing (BDPT) works. How does it differ from unidirectional path tracing? When is it particularly effective?

**Answer:**
- **BDPT:**
  1. Build eye subpath: x₀ (camera) → x₁ → ... → xₖ
  2. Build light subpath: y₀ (light) → y₁ → ... → yₗ
  3. Connect subpaths: Connect xₖ to yₗ
  4. Weight contributions using MIS

- **Differences from unidirectional:**
  - Unidirectional: Only builds paths from camera
  - BDPT: Builds paths from both camera and light
  - BDPT: Evaluates multiple connection strategies

- **Effective for:**
  - Small lights (hard to hit from camera)
  - Caustics (focused light paths)
  - Complex illumination

#### Question 8.2
**Question:** For a path with k=3 eye vertices and l=2 light vertices, how many connection strategies are possible? Explain how MIS weights are computed for BDPT.

**Answer:**
- **Path length:** k=3 means 4 vertices total (including camera)
- **Connection strategies:** s = 0,1,2,3,4 (connect after 0,1,2,3,4 eye vertices)
- **Total strategies:** 5 strategies

- **MIS weights:**
  - For each strategy s, compute probability p_s,t to sample path with that strategy
  - Use balance heuristic: w_s = p_s,t / Σ_i p_i
  - Need to compute probabilities for all strategies that could generate the same path

---

## Topic 9: Participating Media and Subsurface Scattering

#### Question 9.1
**Question:** Write the volume rendering equation. Explain each term: transmittance, extinction, in-scattering, emission.

**Answer:**
- **Volume rendering equation:**
  ```
  L(x,ω) = L_0(x₀,ω) T(x₀→x) + ∫_0^d L_s(x',ω) T(x'→x) dt
  ```

- **Terms:**
  - **L_0:** Radiance entering volume
  - **T(x₀→x):** Transmittance (fraction of light surviving)
  - **L_s:** Source term = L_e + σ_s ∫ p(ω'→ω) L_i(ω') dω'
  - **σ_s:** Scattering coefficient
  - **p:** Phase function

- **Transmittance:**
  ```
  T(x→y) = exp(-∫_x^y σ_t(s) ds)
  ```
  Where σ_t = σ_a + σ_s (extinction = absorption + out-scattering)

#### Question 9.2
**Question:** Light travels through fog with extinction coefficient σ_t = 0.1 m⁻¹. What fraction survives after 10 meters? What distance gives 50% transmittance?

**Answer:**
- **After 10 meters:**
  ```
  T = exp(-0.1 × 10) = exp(-1) ≈ 0.368
  ```
  **Answer:** 36.8% survives

- **50% transmittance:**
  ```
  0.5 = exp(-0.1 × d)
  -ln(0.5) = 0.1 × d
  d = 0.693 / 0.1 = 6.93 meters
  ```
  **Answer:** 6.93 meters

#### Question 9.3
**Question:** Explain the difference between BRDF and BSSRDF. When is BSSRDF needed?

**Answer:**
- **BRDF:**
  - f_r(x, ω_i → ω_o) where light enters and exits at same point x
  - Local reflection model

- **BSSRDF:**
  - S(x_i, ω_i, x_o, ω_o) where light enters at x_i and exits at x_o (different points)
  - Non-local scattering model

- **When needed:**
  - Subsurface scattering (skin, marble, milk)
  - Light enters material, scatters internally, exits elsewhere
  - Cannot be modeled with BRDF

---

---

## Topic 10: Neural Denoising

#### Question 10.1
**Question:** Explain how neural networks are used for denoising Monte Carlo renders. What auxiliary buffers are typically used and why?

**Answer:**
- **Approach:**
  - Input: Noisy rendered image (low samples)
  - Output: Denoised image
  - Training: Minimize loss between denoised output and ground truth (high samples)

- **Auxiliary buffers (AOVs):**
  - Surface normals: Help identify surface orientation
  - Depth: Help identify scene structure
  - Albedo: Separate direct/indirect lighting
  - Direct/indirect separation: Help network understand light transport

- **Why useful:**
  - Network can learn scene structure
  - Better denoising than image-only approaches
  - Preserves details while reducing noise

#### Question 10.2
**Question:** A Monte Carlo renderer produces images with variance σ² = 100. A neural denoiser reduces this to σ² = 4. How many additional samples would be needed for the same variance reduction without denoising?

**Answer:**
- Variance reduction: 100/4 = 25×
- For Monte Carlo: variance ∝ 1/N
- To reduce variance by 25×, need 25× more samples
- **Answer:** 25× more samples needed

---

## Topic 11: Neural Radiosity

#### Question 11.1
**Question:** Explain traditional radiosity and its limitations. How does neural radiosity address these limitations?

**Answer:**
- **Traditional radiosity:**
  - Solves rendering equation for diffuse surfaces only
  - Discretizes scene into mesh faces
  - Solves linear system: B = E + F·B
  - **Limitations:**
    - High mesh resolution needed
    - Large linear systems
    - Cannot handle general BRDFs (requires 4D discretization)

- **Neural radiosity:**
  - Represents radiosity B(x) using neural network
  - Continuous representation (no mesh)
  - Can handle general BRDFs
  - **Advantages:**
    - Efficient storage
    - Adaptive representation
    - Leverages GPU infrastructure

#### Question 11.2
**Question:** Traditional radiosity discretizes a scene into 10,000 mesh faces. If extending to general BRDFs requires 10 samples per dimension for direction (2D), how many elements would be needed?

**Answer:**
- 2D position: 10,000 faces
- 2D direction: 10 × 10 = 100 samples
- Total: 10,000 × 100 = **1,000,000 elements**
- This demonstrates why neural representations are attractive!

---

## Topic 12: Neural Importance Sampling (Primary Sample Space)

#### Question 12.1
**Question:** Explain Primary Sample Space (PSS) importance sampling. What are the advantages over standard importance sampling?

**Answer:**
- **PSS:**
  - Paths constructed from random numbers y ∈ [0,1]^d
  - Mapping: y → x = Φ(y) (path via ray tracing)
  - Learn transformation T: [0,1]^d → [0,1]^d to warp samples

- **Advantages:**
  1. Samples entire paths (not individual bounces)
  2. Handles occlusions (visibility)
  3. Handles non-local effects (caustics)
  4. Unified approach (treats renderer as black box)
  5. Scene-specific adaptation

- **Training:**
  - Minimize variance of Monte Carlo estimator
  - Learn transformation T using neural network
  - Use automatic differentiation

---

---

## Topic 13: 3D Geometry Processing

#### Question 13.1
**Question:** Compare point clouds, polygon meshes, and implicit surfaces as 3D shape representations. What are the advantages and disadvantages of each?

**Answer:**
- **Point Clouds:**
  - Array of 3D points
  - **Advantages:** Simple, raw scanner output
  - **Disadvantages:** Not continuous, no topology

- **Polygon Meshes:**
  - Connected polygons (vertices, edges, faces)
  - **Advantages:** Continuous, explicit topology, easy to render
  - **Disadvantages:** Requires proper connectivity

- **Implicit Surfaces:**
  - Defined by f(x,y,z) = 0
  - **Advantages:** Easy to combine, watertight, smooth
  - **Disadvantages:** Harder to render directly, need isosurface extraction

#### Question 13.2
**Question:** A point cloud contains 1,000,000 points. If converted to a triangle mesh with average 6 triangles per vertex, approximately how many triangles?

**Answer:**
- Each triangle has 3 vertices
- Average 6 triangles per vertex means each vertex shared by ~6 triangles
- Triangles = (vertices × triangles_per_vertex) / vertices_per_triangle
- Triangles = (1,000,000 × 6) / 3 = **2,000,000 triangles**

---

## Topic 14: Scene Reconstruction (NeRF)

#### Question 14.1
**Question:** Explain the NeRF (Neural Radiance Fields) approach to scene reconstruction. Write the volumetric rendering equation used in NeRF.

**Answer:**
- **NeRF:**
  - Represents scene as neural radiance field
  - Network: (x, d) → (σ, c) where σ is density, c is radiance
  - Differentiable rendering enables optimization

- **Volumetric rendering:**
  ```
  C(r) = ∫_0^d T(t) σ(r(t)) c(r(t), d) dt
  ```
  Where:
  - T(t) = exp(-∫_0^t σ(r(s)) ds): Transmittance
  - σ: Density (scattering coefficient)
  - c: Radiance (volumetric emission)

- **Numerical approximation:**
  ```
  C(r) ≈ Σ_i T_i (1 - exp(-σ_i δ)) c_i
  ```
  Where T_i = exp(-Σ_{j=0}^{i-1} σ_j δ)

#### Question 14.2
**Question:** In NeRF, positional encoding γ(p) uses L=10 frequency levels for 3D positions. How many dimensions does the encoded position vector have?

**Answer:**
- Per frequency level: sin and cos = 2 components
- Total per coordinate: L × 2 = 10 × 2 = 20 dimensions
- For 3D position (x,y,z): 3 × 20 = **60 dimensions total**

#### Question 14.3
**Question:** Explain the optimization process for NeRF. What is the loss function? How are gradients computed?

**Answer:**
- **Loss function:**
  ```
  L = Σ_r ||C(r) - Ĉ_coarse(r)||² + ||C(r) - Ĉ_fine(r)||²
  ```
  Where C(r) is ground truth, Ĉ is rendered estimate

- **Optimization:**
  - Gradient descent on network weights
  - Gradients: ∂L/∂θ computed via automatic differentiation
  - Backpropagation through rendering function

- **Hierarchical sampling:**
  - Coarse network (64 samples) → define PDF
  - Fine network (128 samples) → importance sample according to PDF

---

---

## Topic 15: Deep Learning Architectures for 3D Shapes

#### Question 15.1
**Question:** Compare 3D CNNs on voxel grids, PointNet, and Graph Convolutional Networks for processing 3D shapes. When would you use each?

**Answer:**
- **3D CNNs on Voxel Grids:**
  - Discretize shape on 3D grid
  - Perform 3D convolution
  - **Advantages:** Straightforward, regular structure
  - **Disadvantages:** Memory intensive, not rotation invariant
  - **Use when:** Simple shapes, memory not a concern

- **PointNet:**
  - Process point clouds directly
  - Point-wise MLP + max pooling
  - **Advantages:** Permutation invariant, handles variable points
  - **Disadvantages:** No local structure modeling
  - **Use when:** Point cloud data, global features needed

- **Graph Convolutional Networks:**
  - Represent mesh as graph
  - Convolutions on graph structure
  - **Advantages:** Works directly on meshes, preserves topology
  - **Disadvantages:** More complex implementation
  - **Use when:** Mesh data, local features important

#### Question 15.2
**Question:** A 3D shape is discretized into a 128×128×128 voxel grid. If each voxel stores one float (4 bytes), how much memory? Compare to a triangle mesh with 10,000 vertices and 20,000 triangles.

**Answer:**
- **Voxel grid:**
  - 128³ × 4 bytes = 2,097,152 × 4 = **8,388,608 bytes ≈ 8 MB**

- **Mesh:**
  - Vertices: 10,000 × 3 × 4 = 120,000 bytes
  - Triangles: 20,000 × 3 × 4 = 240,000 bytes
  - Total: **360,000 bytes ≈ 0.36 MB**

- **Mesh is much more memory-efficient!**

---

## Topic 16: Generative Modeling for Shapes

#### Question 16.1
**Question:** Explain how Variational Autoencoders (VAEs) work for 3D shape generation. What is the latent space? How do you generate new shapes?

**Answer:**
- **Architecture:**
  - Encoder: Shape → latent code z (lower-dimensional)
  - Decoder: Latent code z → Shape

- **Training:**
  - Reconstruction loss: ||x - decode(encode(x))||²
  - Regularization: KL divergence to prior N(0,I)

- **Latent space:**
  - Lower-dimensional representation of shapes
  - Captures shape variations
  - Continuous space where similar shapes are close

- **Generation:**
  - Sample z ~ N(0,I)
  - Decode: shape = decode(z)

#### Question 16.2
**Question:** A VAE has latent space dimension d=128. If each dimension is quantized to 8 bits, how many distinct shapes can be represented?

**Answer:**
- Each dimension: 2⁸ = 256 values
- Total combinations: 256¹²⁸ = 2¹⁰²⁴
- **Answer:** 2¹⁰²⁴ (theoretically, though many may not be valid shapes)

---

## Topic 17: Shape Synthesis via 2D Generative Modeling

#### Question 17.1
**Question:** Explain how 2D generative models can be used for 3D shape synthesis. Describe at least two approaches.

**Answer:**
- **Approach 1: Multi-View Generation**
  - Generate multiple 2D views using 2D GAN
  - Reconstruct 3D from views
  - Ensure multi-view consistency

- **Approach 2: 2D Parameterization**
  - Flatten 3D shape to 2D domain (UV mapping)
  - Generate 2D texture/geometry maps
  - Map back to 3D

- **Approach 3: Image-to-3D**
  - Generate 2D image (using 2D GAN)
  - Predict 3D shape from image

- **Advantages:**
  - Leverage large 2D image datasets
  - Powerful 2D generators available

- **Challenges:**
  - Multi-view consistency
  - Parameterization quality
  - Ambiguity (multiple 3D shapes for one image)

---

## Topic 18: Controllable Synthesis

#### Question 18.1
**Question:** Explain conditional generation for 3D shapes. How would you modify a VAE to generate shapes conditioned on a class label?

**Answer:**
- **Conditional VAE:**
  - Input: Condition c (label, text, etc.) + latent code z
  - Output: Shape conditioned on c
  - Architecture: Encoder: (x, c) → z, Decoder: (z, c) → x

- **Training:**
  - Paired data: (shape, condition)
  - Learn p(shape | condition)

- **Generation:**
  - Sample z ~ N(0,I)
  - Provide condition c
  - Generate: shape = decode(z, c)

#### Question 18.2
**Question:** A conditional VAE generates chairs. The latent space has dimension 64, and the condition (chair type: "office", "dining", "armchair") is one-hot encoded (3 dimensions). What is the total input dimension to the decoder?

**Answer:**
- Latent code z: 64 dimensions
- Condition c: 3 dimensions (one-hot)
- Total input: 64 + 3 = **67 dimensions**

---

## Comprehensive Review Questions

### Question C.1
**Question:** Derive the rendering equation from first principles, starting from the BRDF definition and energy conservation.

**Answer:**
1. **BRDF definition:**
   ```
   f_r = dL_o / (L_i cos θ_i dω_i)
   ```
   Therefore: dL_o = f_r L_i cos θ_i dω_i

2. **Integrate over hemisphere:**
   ```
   L_o = ∫_hemisphere f_r L_i cos θ_i dω_i
   ```

3. **Add emission:**
   ```
   L_o = L_e + ∫_hemisphere f_r L_i cos θ_i dω_i
   ```

4. **Recognize recursion (L_i comes from L_o at other points):**
   ```
   L_o(x,ω_o) = L_e(x,ω_o) + ∫_hemisphere f_r(x,ω_i→ω_o) L_o(x',ω_i) cos θ_i dω_i
   ```
   Where x' is point where ray (x,ω_i) hits surface.

### Question C.2
**Question:** Compare path tracing, bidirectional path tracing, and photon mapping. When would you use each?

**Answer:**
- **Path Tracing:**
  - Build paths from camera
  - Connect to light at each vertex
  - **Use when:** General scenes, reference implementation

- **Bidirectional Path Tracing:**
  - Build paths from both camera and light
  - Connect in middle
  - **Use when:** Small lights, caustics, complex illumination

- **Photon Mapping:**
  - Shoot photons from light
  - Store in photon map
  - Estimate radiance using density estimation
  - **Use when:** Caustics, participating media

### Question C.3
**Question:** Explain the relationship between Monte Carlo integration, importance sampling, and the rendering equation. How do they work together?

**Answer:**
- **Monte Carlo integration:**
  - Numerical method to estimate integrals
  - Î = (1/N) Σ f(X_i)/p(X_i)

- **Importance sampling:**
  - Choose PDF p proportional to integrand f
  - Reduces variance

- **Rendering equation:**
  - Integral equation for light transport
  - Cannot solve analytically
  - Use Monte Carlo to estimate integral

- **Together:**
  - Path tracing uses Monte Carlo to estimate rendering equation
  - Importance sampling (BRDF, light) reduces variance
  - MIS combines multiple strategies

---

## Study Checklist

Before the exam, make sure you can:

- [ ] Derive key equations (rendering equation, BRDF, Monte Carlo estimator)
- [ ] Explain acceleration structures and their trade-offs
- [ ] Calculate radiometric quantities (radiance, irradiance, intensity)
- [ ] Sample from arbitrary PDFs using inversion method
- [ ] Explain path tracing and Russian roulette
- [ ] Derive three-point form from hemispherical form
- [ ] Compute MIS weights
- [ ] Explain BDPT and connection strategies
- [ ] Write volume rendering equation
- [ ] Explain NeRF architecture and optimization
- [ ] Compare 3D shape representations
- [ ] Explain generative models (VAE, GAN) for shapes
- [ ] Sketch diagrams for light transport paths
- [ ] Solve numerical problems involving Monte Carlo, radiometry, transmittance

---

**Good luck with your exam preparation!**

