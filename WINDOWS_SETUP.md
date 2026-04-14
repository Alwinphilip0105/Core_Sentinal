# Windows: Fix PyTorch "DLL initialization failed" (c10.dll)

**PowerShell:** If the project path contains spaces (e.g. `Sem 4`), always quote the path:

`cd "E:\Rutgers_Class\Sem 4\Project\Core_Sentinal"`

If you see:
```text
OSError: [WinError 1114] A dynamic link library (DLL) initialization routine failed. Error loading "...\c10.dll" or one of its dependencies.
```

**Option A – Install Visual C++ Redistributable (if not already)**  
Download and run: **https://aka.ms/vs/17/release/vc_redist.x64.exe**  
Then close and reopen your terminal.

**Option B – Use an older PyTorch (often fixes DLL issues even when VC++ is installed)**

In your project venv:

```powershell
pip uninstall -y torch torchvision
pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cpu
pip install "numpy<2"
```

(PyTorch 2.4 needs NumPy 1.x; `numpy<2` keeps it compatible.) Then run: `python scripts/manual_probe.py`

**For GPU (CUDA)** after PyTorch loads successfully:

```powershell
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```
(Use `cu124` or `cu126` if your `nvidia-smi` shows CUDA 12.4 or 12.6.)
