# CUDA Python

A leírásokban a Numba künyvtár segítségével fogunk GPU kódokat írni CUDA platformra.

## A Numba könyvtár

### Mi az a Numba?
A Numba egy JIT (Just-In-Time) fordító, amely lehetővé teszi, hogy Python kódunkat futásidőben gépi kóddá fordítsuk. A Python nyelvű számításokat – főleg NumPy-alapú tömbműveleteket – optimalizálja, és képes CUDA GPU-n is futtatható kódot generálni, ha rendelkezésre áll egy NVIDIA GPU és a CUDA toolchain.

### Miért Numba?
Python marad, de CUDA-t használ: nem kell C++ vagy CUDA C nyelvet tanulni.

Gyors prototípusfejlesztés GPU-ra.

Kiváló integráció NumPy-val és Python-ökoszisztémával.

Könnyű beépíteni meglévő képfeldolgozó projektekbe.

### Telepítés

``` bash
pip install numba
```

részletesen: https://numba.readthedocs.io/en/stable/user/installing.html

### CUDA támogatás feltétele:

NVIDIA GPU (Compute Capability ≥ 3.0)

Telepített CUDA Toolkit (pl. nvcc --version működjön)

A numba.cuda modul használatához szükség van a megfelelő NVIDIA driverre is.



