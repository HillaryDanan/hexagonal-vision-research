# Exploring Hexagonal Tokenization for Vision Transformers: An Open Research Question

Dr. Hillary Danan  
January 2025

## Abstract

Despite mathematical proofs showing hexagonal packing is 15.47% more efficient than square packing and the universal adoption of hexagonal arrangements in biological vision systems, all existing vision transformers use square patch extraction. This work investigates this unexplored research direction, providing tools and initial measurements to understand why this gap exists and whether it represents an opportunity for improvement.

## 1. Background

### 1.1 Mathematical Foundation

The superiority of hexagonal packing has been rigorously proven:
- Hexagonal density: π/(2√3) ≈ 0.9069 (Thue, 1890; Fejes Tóth, 1940)
- Square density: π/4 ≈ 0.7854
- Improvement: 15.47%

### 1.2 Biological Evidence

Hexagonal arrangements appear throughout vision systems:
- Human retinal cones: 40-50% hexagonal in parafovea
- Insect compound eyes: Universal hexagonal ommatidia
- Cortical columns: Hexagonal-like organization

### 1.3 Prior Work in Hexagonal CNNs

- **HexCNN** (Zhao et al., 2021): 42.2% training time reduction
- **HexaConv** (Hoogeboom et al., 2018): Leverages 6-fold symmetry
- **HexagDLy** (2019): PyTorch tools for hexagonal convolutions

## 2. Research Gap

Our literature review found **zero** papers on hexagonal vision transformers. All major architectures use square patches:
- ViT, DINOv2: 16×16 patches
- I-JEPA: 16×16 patches
- FlexiViT: Variable but square
- EfficientViT: Square patches

## 3. Research Questions

1. Why have hexagonal patches not been explored for transformers?
2. What are the technical barriers to implementation?
3. Could hexagonal tokenization improve efficiency or accuracy?
4. What tasks might benefit most from hexagonal symmetry?

## 4. Initial Investigation

We provide research tools to explore these questions:

```python
# Configuration
Image size: 224×224
Hex radius: 8 pixels
Square patch: 16×16
```

### 4.1 Coverage Analysis

- Hexagonal patches: 561
- Square patches: 196
- Ratio: 2.86× more hexagonal patches

### 4.2 Implementation Challenges

- No hexagonal positional encodings exist
- Standard frameworks assume rectangular tensors
- No optimized hexagonal operations

### 4.3 Preliminary Measurements

- Extraction overhead: ~2× slower (unoptimized)
- Parameter increase: ~2.8× (due to more patches)
- Memory usage: Proportionally higher

## 5. Discussion

The absence of hexagonal vision transformers despite strong theoretical foundations suggests either:

1. Unknown technical barriers
1. Unexplored opportunity
1. Practical constraints outweigh theoretical benefits

This work provides tools to investigate these possibilities.

## 6. Future Directions

1. Develop hexagonal positional encodings
1. Create optimized extraction operations
1. Design hex-aware attention mechanisms
1. Benchmark on standard vision tasks

## 7. Conclusion

Hexagonal vision transformers represent a completely unexplored research direction. While implementation challenges exist, the mathematical optimality and biological prevalence suggest investigation is warranted.

## Code Availability

Research implementation: github.com/HillaryDanan/hexagonal-vision-research

## References

[List of actual papers cited in the work]