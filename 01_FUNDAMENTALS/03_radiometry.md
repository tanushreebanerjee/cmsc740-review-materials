# Lecture 3: Radiometry

**📄 Reference:** [`pdfs/03 Radiometry.pdf`](../pdfs/03%20Radiometry.pdf)

---

## Physical Model of Light

**Geometrical Optics:**
- Light consists of rays (idealized narrow beams)
- Rays carry "spectrum of light" - Spectral Power Distribution (SPD)
- Rays reflect, refract, scatter at material interfaces
- In homogeneous material: rays travel along straight lines
- In vacuum: power (SPD) along ray is constant
- Valid when wavelength << object size (otherwise wave effects like diffraction)

**Color in Computer Graphics:**
- Store only three samples of SPD: Red, Green, Blue (RGB)
- **Why RGB is enough:** Trichromatic color vision
  - Human eye has three types of photoreceptor cells (S, M, L)
  - Each has different absorption curves
  - Three primaries sufficient to match most colors
- **Color spaces:** Quantifying color
  - Many derived from CIE RGB color matching curves
  - Determined using tristimulus experiment
  - Gamut of 3 primaries doesn't cover all distinct colors

**Spectral Radiance:**
- Energy per time, per wavelength, per solid angle, per area
- In practice: assume steady state, measure at discrete wavelengths (R, G, B)
- **Radiance L:** Power per solid angle per area (vector of 3 values for RGB)
- Function of position x and direction ω

## Fundamental Quantities

### 1. Radiant Energy (Q)
- Total energy emitted, transmitted, or received
- Units: Joules (J)

### 2. Radiant Power / Flux (Φ)
- Energy per unit time: Φ = dQ/dt
- Units: Watts (W)

### 3. Irradiance (E)
- Power per unit area **arriving** at a surface
- E = dΦ/dA
- Units: W/m²

**Key Point:** Irradiance decreases with distance squared (inverse square law)

### 4. Radiance (L)
- Power per unit area per unit solid angle
- L = d²Φ / (dA dω cos θ)
- Units: W/(m²·sr)

**Why Radiance is Fundamental:**
- Radiance is **conserved** along a ray in vacuum
- This makes it the natural quantity for light transport

### 5. Radiant Intensity (I)
- Power per solid angle
- I = dΦ/dω
- Units: W/sr
- For isotropic point source: I constant in all directions
- Total power: Φ = 4πI

## Solid Angle

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

## Bidirectional Reflectance Distribution Function (BRDF)

**Definition:**
```
f_r(ω_i → ω_o) = dL_o(ω_o) / (L_i(ω_i) cos θ_i dω_i)
```

**Physical Properties:**
1. **Reciprocity**: f_r(ω_i → ω_o) = f_r(ω_o → ω_i)
2. **Energy Conservation**: ∫_hemisphere f_r(ω_i → ω_o) cos θ_i dω_i ≤ 1

## Rendering Equation (Preview)

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

## Practice Problem

**Question:** A point light source emits 100W uniformly in all directions. What is the irradiance at a point 2 meters away on a surface perpendicular to the light direction?

**Solution:**
- Power: Φ = 100W
- Distance: r = 2m
- Surface area of sphere at distance r: A = 4πr² = 4π(2)² = 16π m²
- Irradiance: E = Φ/A = 100W / (16π m²) ≈ **1.99 W/m²**

**Follow-up:** If the surface is tilted 45° from perpendicular, what is the irradiance?
- E_tilted = E × cos(45°) = 1.99 × 0.707 ≈ **1.41 W/m²**

---

**Previous:** [Lecture 2: Acceleration Structures](02_acceleration_structures.md) | **Next:** [Lecture 4: Monte Carlo Integration](04_monte_carlo_integration.md) | [Back to Index](../../REVIEW_INDEX.md)

