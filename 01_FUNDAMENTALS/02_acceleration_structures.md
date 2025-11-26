# Lecture 2: Acceleration Structures

**📄 Reference:** [`pdfs/02 Acceleration structures.pdf`](../pdfs/02%20Acceleration%20structures.pdf)

---

## Purpose
Accelerate ray-scene intersection queries, which are the computational bottleneck in ray tracing.

## Key Data Structures

### 1. Bounding Volume Hierarchy (BVH)

**Concept:**
- Hierarchical tree structure
- Each node contains a bounding volume (AABB - Axis-Aligned Bounding Box)
- Scene objects are partitioned into smaller groups

**Construction:**
- Recursively partition space
- Split along longest axis
- Stop when node contains few objects (leaf node)

**BVH Construction Pseudocode:**
```
function buildBVH(objects):
    if len(objects) <= MAX_OBJECTS_PER_LEAF:
        return LeafNode(objects)
    
    // Find bounding box
    bbox = computeBoundingBox(objects)
    
    // Split along longest axis
    axis = longestAxis(bbox)
    splitPos = median(objects, axis)  // or other strategy
    
    // Partition objects
    left = [obj for obj in objects if obj.center[axis] < splitPos]
    right = [obj for obj in objects if obj.center[axis] >= splitPos]
    
    // Recursively build children
    return InternalNode(
        bbox,
        buildBVH(left),
        buildBVH(right)
    )
```

**Query:**
- Traverse tree from root
- Test ray against bounding volumes
- Only test objects in intersected nodes

**BVH Query Pseudocode:**
```
function intersectBVH(ray, node):
    // Test ray against bounding box
    if not intersectRayAABB(ray, node.bbox):
        return null
    
    if node.isLeaf():
        // Test all objects in leaf
        closest = null
        for obj in node.objects:
            hit = intersectRayObject(ray, obj)
            if hit and (closest == null or hit.t < closest.t):
                closest = hit
        return closest
    else:
        // Recursively test children
        hitLeft = intersectBVH(ray, node.left)
        hitRight = intersectBVH(ray, node.right)
        return closest(hitLeft, hitRight)
```

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

### 2. Spatial Subdivision

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

**k-d Tree (k-dimensional tree):**
- Binary space partitioning (BSP) tree
- Dividing planes are axis-aligned
- Recursively divide space into two parts using dividing planes
- Cycle through splitting axis from one level to next
- **Construction:** Minimize expected number of intersection tests
- **Traversal:** Front-to-back traversal
  - Traverse child nodes in order (front to back) along rays
  - Stop as soon as first surface intersection found
  - Maintain stack of subtrees (more efficient than recursion)
- **Advantage over octree:** Fewer children per node (2 vs 8) → faster traversal
- Binary space partitioning tree
- Alternates splitting along x, y, z axes
- Creates axis-aligned planes
- **Properties**: More memory efficient than BVH, good for static geometry

### 3. Spatial Data Structures Comparison

| Structure | Construction | Query | Memory | Dynamic Updates |
|-----------|--------------|-------|--------|-----------------|
| BVH | O(n log n) | O(log n) | Medium | Moderate |
| k-d Tree | O(n log² n) | O(log n) | Low | Difficult |
| Grid | O(n) | O(1) avg | High | Easy |

## Practice Problem

**Question:** Given a scene with 1,000,000 triangles, how many intersection tests would be needed without acceleration structures? With a BVH (assuming log₂(1,000,000) ≈ 20 levels)?

**Solution:**
- Without acceleration: Up to 1,000,000 tests per ray
- With BVH: Approximately 20 × (average objects per leaf) tests
- If leaf nodes contain ~10 objects: ~200 tests per ray
- **Speedup: ~5,000x** (theoretical, actual depends on tree quality)

---

**Previous:** [Lecture 1: Introduction](01_introduction_ray_tracing.md) | **Next:** [Lecture 3: Radiometry](03_radiometry.md) | [Back to Index](../../REVIEW_INDEX.md)

