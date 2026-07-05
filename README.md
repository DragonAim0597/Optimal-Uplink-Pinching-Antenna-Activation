# Optimal Uplink Pinching-Antenna Activation

MATLAB code for reproducing the numerical results in:

> Z. Cheng and C. Ouyang, "Optimal Uplink Pinching-Antenna Activation."

The code evaluates discrete pinching-antenna (PA) activation in an uplink
multiuser pinching-antenna system. One PA location is selected from a finite
candidate set on each dielectric waveguide. The implemented methods are:

- Greedy search (GS)
- Beam search (BeS)
- Globally optimal branch-and-bound search (BnB)
- A centered half-wavelength fixed array used as a benchmark

## Requirements

- MATLAB R2020b or later
- Parallel Computing Toolbox only if `useParallel = true` is selected in the
  Fig. 4 script

No external datasets or third-party MATLAB packages are required.

## Reproducing the figures

Run each script directly from MATLAB. All configurable parameters are grouped
at the beginning of each file.

| Paper figure | MATLAB script |
| --- | --- |
| Fig. 3(a): sum-rate versus transmit power | `Fig3a_Sum_Rate_vs_Transmit_Power.m` |
| Fig. 3(b): sum-rate versus service-region length | `Fig3b_Sum_Rate_vs_Dx.m` |
| Fig. 4: performance and runtime versus candidate count | `Fig4_Performance_Runtime_vs_Number_of_Candidates.m` |
| Fig. 5(a): sum-rate versus beam width | `Fig5a_Sum_Rate_vs_Beam_Width.m` |
| Fig. 5(b): runtime versus beam width | `Fig5b_Runtime_vs_Beam_Width.m` |

Each script:

1. Creates the `results` directory if needed.
2. Validates BnB against exhaustive search on a small random instance.
3. Runs the Monte Carlo experiment.
4. Saves checkpoints, final `.mat` data, and figure files in `results`.

The default configuration uses 1000 independent user deployments, as stated
in the paper. For a quick code check, set `numTrials` to `10` or `20` and set
`resumeFromCheckpoint = false`. Delete an incompatible checkpoint before
changing simulation parameters.

## Main simulation parameters

- Carrier frequency: 28 GHz
- Number of users and waveguides: `K = M = 4`
- Noise power: -90 dBm
- Waveguide attenuation: 0.08 dB/m
- Effective refractive index: 1.4
- Waveguide height: 3 m

The spatial channel includes the free-space factor
`sqrt(eta) = c/(4*pi*fc)`, spherical phase, distance-dependent attenuation,
guided-wave phase, and in-waveguide attenuation.

## Runtime note

Fig. 4 can be computationally demanding because BnB has exponential
worst-case complexity and the largest setting uses 151 candidates per
waveguide. Checkpointing is enabled by default.

## Citation

If this code is useful in your research, please cite the associated paper.
The final bibliographic information can be added here after publication.

