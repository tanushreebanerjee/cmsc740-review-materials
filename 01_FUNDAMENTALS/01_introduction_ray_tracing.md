# Lecture 1: Introduction to Realistic Rendering

**📄 Reference:** [`pdfs/01 Introduction.pdf`](../pdfs/01%20Introduction.pdf)

---

## Key Concepts

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

## Camera Setup

**Camera Coordinate System:**
- User parameters: `from`, `to`, `up` vectors
- Construct basis (u, v, w) for camera coordinates
- Extrinsic 4×4 matrix [u v w e] transforms from world to camera coordinates
- Right-handed coordinate system

**Viewing Frustum:**
- Pyramidal volume containing 3D volume seen by camera
- Defined by:
  - Vertical field-of-view θ
  - Aspect ratio = width/height
  - Image plane at w = -1 (by convention)

**Primary Ray Computation:**
1. **Intrinsic matrix K:** Maps pixel (i,j) to ray in camera coordinates
   - Pixel (i,j) → point (u,v) on image plane
   - Primary ray: p(t) = (0,0,0) + t·(u, v, -1) in camera coordinates
2. **Extrinsic matrix:** Transform ray to world coordinates
   - p_world(t) = [u v w e] · p_camera(t)

**Image Plane Coordinates:**
- Image resolution: m × n pixels
- Pixel (i,j) center: u = l + (r-l)(i+0.5)/m, v = b + (t-b)(j+0.5)/n
- Bounds: t, b, l, r computed from field-of-view and aspect ratio
- Primary rays can go through pixel centers or random positions within pixels

**Ray-Surface Intersection:**
- **Implicit surfaces**: f(x,y,z) = 0, solve f(p(t)) = 0
- **Parametric surfaces**: p = (x(u,v), y(u,v), z(u,v)), solve ray = surface
- **Triangles**: Use Cramer's rule, check barycentric coordinates

**Surface Normals:**
- **Implicit**: n = ∇f / ||∇f||
- **Parametric**: n = (∂p/∂u × ∂p/∂v) / ||∂p/∂u × ∂p/∂v||

## Important Terms

- **Global Illumination**: Accounting for all light interactions (direct + indirect)
- **Path Tracing**: Technique for sampling light paths
- **Ray Tracing**: Geometric technique for finding intersections
- **Soft shadows, caustics, indirect illumination**: Effects requiring global illumination

---

**Next:** [Lecture 2: Acceleration Structures](02_acceleration_structures.md) | [Back to Index](../../REVIEW_INDEX.md)

