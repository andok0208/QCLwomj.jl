# QCLwomj.jl
Juila code for mixed quantum classical Liouville molecular dynamics simulation (without momentum jump).

## Quick start
- clone the repository and run:
```bash
cd QCLwomj.jl/src/
julia main.jl
```

## Reference
This code is a GPU implementation of the Fortran code developed in

K. Ando and M. Santer, "Mixed quantum-classical Liouville molecular dynamics without momentum jump", *J. Chem. Phys.* **118**, 10399-10406 (2003).

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

