# CMSC 740 Final Exam Study Plan

**Final Exam Date:** Thursday, December 18, 10:30am - 12:30pm, IRB2107  
**Time Remaining:** ~2 weeks

---

## Study Strategy Overview

### Exam Format
- Questions require **text answers, equations, and sketches**
- Some questions involve **calculations** based on example problems
- **No multiple choice questions** typically
- Be prepared to:
  - Derive equations from first principles
  - Explain concepts clearly
  - Draw diagrams
  - Solve numerical problems
  - Compare and contrast different techniques

### Study Approach
1. **Understand fundamentals first** - Build strong foundation
2. **Practice derivations** - Be able to derive key equations
3. **Work through problems** - Practice calculations and sketches
4. **Review connections** - Understand how topics relate
5. **Test yourself** - Use question bank to identify weak areas

---

## 14-Day Study Schedule

### Week 1: Core Rendering Fundamentals (Days 1-7)

#### Day 1: Introduction & Ray Tracing Basics
**Time:** 2-3 hours
- [ ] Read: Part 1 - Introduction section
- [ ] Review: Ray tracing pseudocode
- [ ] Practice: Question Bank 1.1.1, 1.1.2, 1.1.3
- [ ] Focus: Understand rendering pipeline vs. ray tracing
- [ ] Sketch: Draw ray tracing diagram

**Key Concepts:**
- Rendering pipeline vs. ray tracing
- Ray-surface intersection
- Camera setup and primary rays

#### Day 2: Acceleration Structures
**Time:** 2-3 hours
- [ ] Read: Part 1 - Acceleration Structures section
- [ ] Practice: Question Bank 1.2.1, 1.2.2, 1.2.3
- [ ] Focus: BVH construction and query
- [ ] Calculate: Complexity analysis problems

**Key Concepts:**
- BVH, k-d trees, spatial subdivision
- Time complexity: O(n log n) construction, O(log n) query
- When to use each structure

#### Day 3: Radiometry Fundamentals
**Time:** 3-4 hours
- [ ] Read: Part 1 - Radiometry section
- [ ] Practice: Question Bank 1.3.1, 1.3.2, 1.3.3, 1.3.4
- [ ] Focus: Radiance, irradiance, intensity definitions
- [ ] Derive: Why radiance is conserved along rays
- [ ] Calculate: Irradiance problems

**Key Concepts:**
- Radiance L(x,ω): d²Φ/(dA dω cos θ)
- Irradiance E(x): ∫ L_i cos θ_i dω_i
- Radiance conservation in vacuum
- Solid angle integration

#### Day 4: Monte Carlo Integration - Theory
**Time:** 3-4 hours
- [ ] Read: Part 1 - Monte Carlo Integration section
- [ ] Practice: Question Bank 1.4.1, 1.4.2
- [ ] Focus: Unbiased estimator, variance
- [ ] Derive: Monte Carlo estimator from first principles
- [ ] Calculate: Variance reduction examples

**Key Concepts:**
- Monte Carlo estimator: Î = (1/N) Σ f(X_i)/p(X_i)
- Unbiased: E[Î] = I
- Variance: Var[Î] = (1/N) Var[f(X)/p(X)]
- Convergence: O(1/√N)

#### Day 5: Monte Carlo Integration - Sampling
**Time:** 3-4 hours
- [ ] Read: Part 1 - Sampling PDFs section
- [ ] Practice: Question Bank 1.4.3, 1.4.4, 1.4.5
- [ ] Focus: Inversion method, 2D sampling
- [ ] Practice: Sample from various PDFs
- [ ] Understand: Stratified sampling, QMC

**Key Concepts:**
- Inversion method: P⁻¹(U) where U ~ Uniform[0,1]
- 2D sampling: Marginal and conditional PDFs
- Sample warping: Transform between coordinate systems
- Stratified sampling, quasi-Monte Carlo

#### Day 6: BRDF and Reflection Integral
**Time:** 3-4 hours
- [ ] Read: Part 2 - BRDF section
- [ ] Practice: Question Bank 2.1.1, 2.1.2, 2.1.3, 2.1.4
- [ ] Focus: BRDF definition, properties, models
- [ ] Derive: Reflection integral from BRDF
- [ ] Calculate: Lambertian reflection examples

**Key Concepts:**
- BRDF: f_r = dL_o / (L_i cos θ_i dω_i)
- Properties: Reciprocity, energy conservation, positivity
- Models: Lambertian, specular, Torrance-Sparrow

#### Day 7: Review Week 1 + Practice
**Time:** 3-4 hours
- [ ] Review: All Week 1 topics
- [ ] Practice: Comprehensive questions from Question Bank
- [ ] Identify: Weak areas to review
- [ ] Test: Derive key equations without notes
- [ ] Sketch: Draw diagrams for all major concepts

---

### Week 2: Advanced Topics & Neural Methods (Days 8-14)

#### Day 8: Rendering Equation & Path Tracing
**Time:** 3-4 hours
- [ ] Read: Part 2 - Rendering Equation section
- [ ] Practice: Question Bank 2.2.1, 2.2.2, 2.2.3, 2.2.4
- [ ] Focus: Operator form, Neumann series
- [ ] Derive: Rendering equation from BRDF
- [ ] Code: Path tracing pseudocode

**Key Concepts:**
- Rendering equation: L = L_e + T L
- Solution: L = Σ T^k L_e (Neumann series)
- Path tracing: Monte Carlo solution
- Russian roulette: Probabilistic termination

#### Day 9: Advanced Sampling & BDPT
**Time:** 3-4 hours
- [ ] Read: Part 2 - Advanced Sampling, BDPT sections
- [ ] Practice: Question Bank 2.3.1, 2.3.2, 2.4.1, 2.4.2
- [ ] Focus: MIS, surface form, BDPT
- [ ] Calculate: MIS weight examples
- [ ] Understand: Connection strategies in BDPT

**Key Concepts:**
- Multiple Importance Sampling (MIS)
- Balance heuristic: w_i = p_i / Σ p_j
- Surface form: Integration over surface area
- BDPT: Eye + light subpaths

#### Day 10: Participating Media
**Time:** 2-3 hours
- [ ] Read: Part 2 - Participating Media section
- [ ] Practice: Question Bank 2.5.1, 2.5.2, 2.5.3
- [ ] Focus: Volume rendering equation
- [ ] Calculate: Transmittance problems
- [ ] Understand: BSSRDF vs. BRDF

**Key Concepts:**
- Volume rendering equation
- Transmittance: T = exp(-∫ σ_t ds)
- Extinction: σ_t = σ_a + σ_s
- Subsurface scattering: BSSRDF

#### Day 11: Neural Rendering
**Time:** 3-4 hours
- [ ] Read: Part 3 - All sections
- [ ] Practice: Question Bank 3.1.1, 3.1.2, 3.2.1, 3.2.2, 3.3.1
- [ ] Focus: Denoising, neural radiosity, PSS sampling
- [ ] Understand: Advantages of neural representations
- [ ] Compare: Traditional vs. neural methods

**Key Concepts:**
- Neural denoising: Input noisy image → clean image
- Neural radiosity: Continuous representation
- PSS importance sampling: Learn to sample paths

#### Day 12: Geometry Processing & NeRF
**Time:** 3-4 hours
- [ ] Read: Part 4 - All sections
- [ ] Practice: Question Bank 4.1.1, 4.1.2, 4.2.1, 4.2.2, 4.2.3
- [ ] Focus: Shape representations, NeRF
- [ ] Derive: NeRF volumetric rendering
- [ ] Calculate: Memory/complexity comparisons

**Key Concepts:**
- Point clouds, meshes, implicit surfaces
- NeRF: Neural radiance fields
- Volumetric rendering: C(r) = ∫ T σ c dt
- Positional encoding

#### Day 13: Data-Driven Shape Modeling
**Time:** 3-4 hours
- [ ] Read: Part 5 - All sections
- [ ] Practice: Question Bank 5.1.1, 5.1.2, 5.2.1, 5.2.2, 5.3.1, 5.4.1, 5.4.2
- [ ] Focus: Deep learning architectures, generative models
- [ ] Understand: VAE, GAN, conditional generation
- [ ] Compare: Different shape representations for DL

**Key Concepts:**
- 3D CNNs, PointNet, Graph CNNs
- VAE: Encoder-decoder with latent space
- Generative modeling: VAE, GAN, diffusion
- 2D generative modeling for 3D

#### Day 14: Final Review & Practice
**Time:** 4-5 hours
- [ ] Review: All comprehensive questions (Question Bank C.1, C.2, C.3)
- [ ] Practice: Derive all key equations from memory
- [ ] Test: Work through question bank problems
- [ ] Review: Study checklist (end of Question Bank)
- [ ] Final: Identify and review weak areas

**Key Activities:**
- Derive: Rendering equation, BRDF, Monte Carlo estimator
- Sketch: Light transport paths, acceleration structures
- Calculate: Radiometry, transmittance, MIS weights
- Explain: All major algorithms and their trade-offs

---

## Daily Study Routine

### Recommended Schedule (per study day)
1. **Review (30 min):** Read relevant section from review materials
2. **Practice (1-2 hours):** Work through question bank problems
3. **Derive (30-60 min):** Practice deriving key equations
4. **Sketch (15-30 min):** Draw diagrams for concepts
5. **Review (30 min):** Review what you learned, identify questions

### Study Tips

1. **Active Learning:**
   - Don't just read - work through problems
   - Derive equations yourself
   - Explain concepts out loud

2. **Spaced Repetition:**
   - Review previous days' topics briefly each day
   - Build on fundamentals throughout

3. **Problem-Solving:**
   - Work through calculations step-by-step
   - Check your work
   - Understand why, not just how

4. **Visual Learning:**
   - Draw diagrams for all major concepts
   - Sketch light transport paths
   - Visualize data structures

5. **Test Yourself:**
   - Try problems without looking at solutions first
   - Derive equations without notes
   - Explain concepts to yourself

---

## Priority Topics (Must Know)

### Highest Priority
1. **Rendering Equation** - Core of everything
2. **Monte Carlo Integration** - How we solve rendering equation
3. **BRDF** - How surfaces reflect light
4. **Path Tracing** - Main algorithm
5. **Radiometry** - Fundamental quantities

### High Priority
6. **Acceleration Structures** - BVH, complexity
7. **Importance Sampling** - Variance reduction
8. **MIS** - Combining strategies
9. **BDPT** - Advanced path sampling
10. **NeRF** - Modern scene representation

### Medium Priority
11. **Participating Media** - Volume rendering
12. **Neural Rendering** - Denoising, neural radiosity
13. **Geometry Processing** - Shape representations
14. **Generative Modeling** - VAE, GAN for shapes

---

## Equation Cheat Sheet (Memorize These)

### Radiometry
- Radiance: L = d²Φ/(dA dω cos θ)
- Irradiance: E = ∫ L_i cos θ_i dω_i
- Radiance conservation: L along ray is constant (vacuum)

### BRDF
- Definition: f_r = dL_o / (L_i cos θ_i dω_i)
- Reflection: L_o = ∫ f_r L_i cos θ_i dω_i
- Lambertian: f_r = ρ_d / π

### Rendering Equation
- L_o = L_e + ∫ f_r L_i cos θ_i dω_i
- Operator: L = L_e + T L
- Solution: L = Σ T^k L_e

### Monte Carlo
- Estimator: Î = (1/N) Σ f(X_i)/p(X_i)
- Variance: Var[Î] = (1/N) Var[f(X)/p(X)]

### Transmittance
- T = exp(-∫ σ_t ds)
- Beer's law: T = exp(-σ_t d) for homogeneous

---

## Final Exam Day Checklist

### Before the Exam
- [ ] Get good sleep (8 hours)
- [ ] Eat a good breakfast
- [ ] Bring: Calculator, pencils, eraser
- [ ] Arrive 10 minutes early
- [ ] Review key equations one last time

### During the Exam
- [ ] Read all questions first
- [ ] Start with questions you know best
- [ ] Show all work for calculations
- [ ] Draw clear diagrams
- [ ] Derive equations step-by-step
- [ ] Check units in calculations
- [ ] Manage time (2 hours total)

### Exam Strategy
1. **Quick scan (5 min):** Read all questions
2. **Easy questions first (30 min):** Build confidence
3. **Medium questions (60 min):** Main work
4. **Hard questions (20 min):** Partial credit
5. **Review (5 min):** Check work, fill gaps

---

## Resources

### Review Materials
- [ ] Part 1: Rendering Fundamentals
- [ ] Part 2: Advanced Rendering Techniques
- [ ] Part 3: Neural Rendering
- [ ] Part 4: Geometry Processing & Scene Reconstruction
- [ ] Part 5: Data-Driven Shape Modeling
- [ ] Question Bank (organized by topic)
- [ ] Equation Reference Sheet

### Practice
- [ ] Work through all question bank problems
- [ ] Derive key equations from memory
- [ ] Draw diagrams for all major concepts
- [ ] Solve calculation problems

---

## Study Progress Tracker

### Week 1 Progress
- [ ] Day 1: Introduction & Ray Tracing
- [ ] Day 2: Acceleration Structures
- [ ] Day 3: Radiometry
- [ ] Day 4: Monte Carlo Theory
- [ ] Day 5: Monte Carlo Sampling
- [ ] Day 6: BRDF
- [ ] Day 7: Week 1 Review

### Week 2 Progress
- [ ] Day 8: Rendering Equation
- [ ] Day 9: Advanced Sampling & BDPT
- [ ] Day 10: Participating Media
- [ ] Day 11: Neural Rendering
- [ ] Day 12: Geometry & NeRF
- [ ] Day 13: Shape Modeling
- [ ] Day 14: Final Review

---

**Good luck with your exam preparation! You've got this! 🚀**

