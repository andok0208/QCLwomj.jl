# QCLwomj.jl
Juila code for mixed quantum classical Liouville molecular dynamics simulation (without momentum jump).

## Quick start
Clone the repository and run:
```bash
cd QCLwomj.jl/cuda_nospawn/tests/
julia main_gpu.jl
```
The results (`ptHad.dat` etc.) are in `outdat/` directory.

## Reference
This code is a GPU implementation of the Fortran code developed in:
<br>
K. Ando and M. Santer, "Mixed quantum-classical Liouville molecular dynamics without momentum jump", [*J. Chem. Phys.* **118**, 10399-10406 (2003)](https://dx.doi.org/10.1063/1.1574015).

```
@article{Ando2003_QCL,
  author  = {Ando, Koji and Santer, Mark},
  title   = {Mixed quantum-classical {Liouville} molecular dynamics without momentum jump},
  journal = {J. Chem. Phys.},
  volume  = {118},
  number  = {23},
  pages   = {10399--10406},
  year    = {2003},
  doi     = {10.1063/1.1574015}
}
```
The GPU implementation is described in:
<br>
K. Ando, "GPU implementation of mixed quantum-classical Liouville molecular dynamics without momentum jump", [arXiv.2608.14544](https:arxiv.org/abs/2608.14544)

