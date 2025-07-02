# Hexagonal Vision Transformer Research

An exploration of why vision transformers don't use hexagonal patches despite mathematical and biological evidence for their superiority.

## Research Motivation

- **Mathematical fact**: Hexagonal packing is 15.47% more efficient than square packing
- **Biological fact**: Vision systems universally use hexagonal arrangements  
- **Curious gap**: No research exists on hexagonal vision transformers

## What This Is

**Research tools** to investigate an open question. We provide:

1. **Theoretical analysis** comparing hexagonal vs square tokenization
2. **Measurement tools** for extraction efficiency and coverage
3. **Research implementation** (not optimized for production)
4. **Identified challenges** and research directions

## What This Is Not

- Not a production-ready system
- Not claiming performance improvements
- Not a solved problem
- Not optimized code

## Quick Start

```bash
# Clone repository
git clone https://github.com/HillaryDanan/hexagonal-vision-research.git
cd hexagonal-vision-research

# Install dependencies
pip install torch numpy matplotlib

# Run research experiments
python src/hexagonal_research.py
```

## Initial Findings

From our preliminary investigation:

1. **Coverage**: Hexagons achieve better theoretical coverage (90.69% vs 78.54%)
1. **Patch count**: ~2.86× more hexagonal patches for same radius
1. **Challenges**: No positional encodings, no optimized operations
1. **Implementation**: Currently ~2× slower due to lack of optimization

## Research Questions

1. Can hexagonal tokenization improve transformer efficiency?
1. How to design hexagonal positional encodings?
1. What tasks benefit from 6-fold rotational symmetry?
1. Why has this not been explored?

## Known Limitations

- Unoptimized implementation (research prototype)
- No trained models for comparison
- Missing key components (positional encoding)
- Framework constraints (rectangular tensors)

## Contributing

This is open research. Contributions welcome:

- Hexagonal positional encoding designs
- Optimized implementations
- Benchmark results
- Theoretical analysis

## Related Work

- **HexCNN** (Zhao et al., 2021): 42% training speedup for CNNs
- **HexaConv** (Hoogeboom et al., 2018): Group equivariant hexagonal convolutions
- **HexagDLy** (2019): PyTorch hexagonal tools

## Citation

```bibtex
@software{danan2025hexvit_research,
  author = {Danan, Hillary},
  title = {Hexagonal Vision Transformer Research},
  year = {2025},
  url = {https://github.com/HillaryDanan/hexagonal-vision-research}
}
```

## Contact

Dr. Hillary Danan  
Email: danan.hillary@gmail.com  
LinkedIn: [/in/hillarydanan](https://linkedin.com/in/hillarydanan)

-----

*This is exploratory research investigating an open problem in computer vision.*