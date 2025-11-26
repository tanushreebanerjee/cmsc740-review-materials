# Lecture 4: Monte Carlo Integration

**📄 Reference:** [`pdfs/04 Monte Carlo integration.pdf`](../pdfs/04%20Monte%20Carlo%20integration.pdf)

---

## Motivation

Many rendering integrals cannot be solved analytically:
- Complex BRDFs
- Complex geometry
- Multiple bounces

Monte Carlo provides a **numerical solution**.

## Basic Monte Carlo Estimator

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

## Importance Sampling

**Goal:** Reduce variance by sampling more where f(x) is large.

**Optimal PDF:** p*(x) = f(x)/I (proportional to integrand)

**Variance reduction:**
- Uniform sampling: Var ∝ ∫ f²(x) dx
- Importance sampling: Var ∝ ∫ f²(x)/p(x) dx

## Example: Integrating a Function

**Problem:** Estimate I = ∫_0^1 x² dx = 1/3

**Uniform Sampling:**
- Sample X_i ~ Uniform[0,1]
- Estimate: Î = (1/N) Σ X_i²
- Variance: Var = (1/N) × (1/5 - 1/9) = 4/(45N)

**Importance Sampling (p(x) = 2x):**
- Sample X_i from p(x) = 2x (using inverse CDF: X = √U)
- Estimate: Î = (1/N) Σ X_i²/(2X_i) = (1/N) Σ X_i/2
- Lower variance!

## Sampling PDFs: The Inversion Method

**Problem:** Generate samples X from arbitrary PDF p(x)

**1D Continuous Case:**
1. Compute CDF: P(x) = ∫_{-∞}^x p(t) dt
2. Compute inverse: P⁻¹(u) where u ~ Uniform[0,1]
3. Sample: X = P⁻¹(U) where U ~ Uniform[0,1]

**Inversion Method Pseudocode (1D):**
```
function sampleFromPDF(p, domain):
    // Precompute CDF
    P = computeCDF(p, domain)
    P_inv = computeInverse(P)
    
    // Sample
    u = randomUniform()
    x = P_inv(u)
    return x
```

**Discrete Case:**
1. Compute discrete CDF: P_i = Σ_{j=0}^i p_j
2. Sample u ~ Uniform[0,1]
3. Find i such that P_{i-1} < u ≤ P_i

**Discrete Sampling Pseudocode:**
```
function sampleDiscrete(probabilities):
    // probabilities = [p_0, p_1, ..., p_{n-1}]
    // Compute CDF
    cdf = [0]
    for i = 1 to n:
        cdf[i] = cdf[i-1] + probabilities[i-1]
    
    // Sample
    u = randomUniform() * cdf[n-1]
    // Binary search to find i where cdf[i-1] < u <= cdf[i]
    return binarySearch(cdf, u)
```

## 2D Sampling (Sample Warping)

**Goal:** Generate samples on desired 2D domain (disk, hemisphere, etc.) with desired density

**Approach:**
1. Start with canonical uniform random variables ξ₁, ξ₂
2. Transform to desired domain and density

**Example: Cosine-weighted hemisphere sampling**
- Uniform samples on disk → map to hemisphere
- Results in p(ω) = cos(θ)/π distribution

**Cosine-Weighted Hemisphere Sampling Pseudocode:**
```
function sampleCosineHemisphere():
    // Sample uniform on disk
    u1 = randomUniform()
    u2 = randomUniform()
    r = sqrt(u1)
    phi = 2 * PI * u2
    x = r * cos(phi)
    y = r * sin(phi)
    
    // Map to hemisphere
    z = sqrt(1 - x*x - y*y)
    return normalize(x, y, z)
```

**Theory:**
- Express PDF in convenient coordinates
- Transform PDF to sampling coordinates using Jacobian
- Use marginal and conditional PDFs: p(x,y) = p(x) p(y|x)
- Sample each 1D PDF using inversion method

**General 2D Sampling Pseudocode:**
```
function sample2D(pdf_2d):
    // Factor into marginal and conditional
    pdf_marginal = integrate(pdf_2d, dim=1)  // p(x)
    pdf_conditional = pdf_2d / pdf_marginal   // p(y|x)
    
    // Sample marginal
    x = sampleFromPDF(pdf_marginal, inversion_method)
    
    // Sample conditional
    y = sampleFromPDF(pdf_conditional(x), inversion_method)
    
    return (x, y)
```

## Variance Reduction Techniques

1. **Importance Sampling**: Sample proportional to integrand
2. **Stratified Sampling**: Divide domain into strata, sample uniformly in each
   - **Jittered sampling**: Random sample in each grid cell
   - **Latin hypercube (N-rooks)**: One sample per row/column
3. **Quasi-Monte Carlo**: Use low-discrepancy sequences (Halton, Sobol)
   - Better convergence: O((log N)^d / N) vs O(1/√N)
   - Deterministic (not random)
4. **Multiple Importance Sampling (MIS)**: Combine multiple sampling strategies (covered in Part 2)

## Practice Problems

### Problem 1
**Question:** Estimate ∫_0^π sin(x) dx = 2 using Monte Carlo with:
a) Uniform sampling on [0,π]
b) Importance sampling with p(x) = (1/2) sin(x)

**Solution:**

**a) Uniform Sampling:**
- Sample X_i ~ Uniform[0,π]
- Estimate: Î = (π/N) Σ sin(X_i)
- Expected value: E[Î] = π × (2/π) = 2 ✓

**b) Importance Sampling:**
- p(x) = (1/2) sin(x) (normalized: ∫_0^π p(x) dx = 1)
- CDF: P(x) = (1/2)(1 - cos(x))
- Inverse: x = arccos(1 - 2u)
- Estimate: Î = (1/N) Σ sin(X_i) / [(1/2) sin(X_i)] = (1/N) Σ 2 = 2 ✓

### Problem 2
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

**Previous:** [Lecture 3: Radiometry](03_radiometry.md) | **Next:** [Part 2: Advanced Rendering](../02_ADVANCED_RENDERING/) | [Back to Index](../../REVIEW_INDEX.md)

