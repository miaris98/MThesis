# CARLA Simulator Hardware & CPU Requirements

## CPU Architecture Compatibility (AVX2 / FMA Requirement)
- **Error Symptom**: `Illegal instruction` (SIGILL) when launching `CarlaUE4.sh` / `CarlaUE4-Linux-Shipping`.
- **Root Cause**: The official CARLA precompiled binaries (Unreal Engine 4) are compiled with required modern vector extensions (**AVX2**, **FMA**, **BMI1/BMI2**). If the host CPU lacks AVX2 or the VM hypervisor (e.g. QEMU/KVM on budget cloud hosts) does not pass through AVX2 CPU flags, the Linux kernel terminates CARLA with `SIGILL (Illegal instruction)`.
- **Remedy**: Switch the rented cloud instance (Vast.ai, RunPod, Lambda) to an instance with an **AVX2-compatible CPU**.

### Recommended CPUs on Vast.ai / Cloud:
- **AMD**: AMD EPYC (7xx2 Rome, 7xx3 Milan, 9xx4 Genoa), AMD Ryzen (5000 / 7000 / 9000 series), Threadripper.
- **Intel**: Intel Xeon Scalable (Gold/Platinum 2nd/3rd/4th Gen), Intel Core i7/i9 (10th Gen or newer).
- **CPUs to Avoid**: Older Xeon E5 v1/v2/v3 or legacy hypervisors without CPU flag passthrough.
