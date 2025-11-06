# Quick Reference Guide

## Activate Virtual Environment
```powershell
.\venv\Scripts\Activate.ps1
```

## Verify Setup
```powershell
python verify_setup.py
```

## Start Jupyter Notebook
```powershell
jupyter notebook
```

## Common Commands

### Install packages
```powershell
pip install package_name
```

### Save current environment
```powershell
pip freeze > requirements.txt
```

### Check Python version
```powershell
python --version
```

### Check installed packages
```powershell
pip list
```

### Deactivate virtual environment
```powershell
deactivate
```

## Git Commands

### Initialize repository
```powershell
git init
git add .
git commit -m "Initial commit"
```

### Create and push to GitHub
```powershell
git remote add origin https://github.com/yourusername/ts-anomaly-detection.git
git branch -M main
git push -u origin main
```

## Project Workflow

1. **Activate environment**: `.\venv\Scripts\Activate.ps1`
2. **Start Jupyter**: `jupyter notebook`
3. **Work on notebooks** in `notebooks/` directory
4. **Save and commit**: `git add . && git commit -m "message"`
5. **Deactivate when done**: `deactivate`

## Troubleshooting

### If activation fails
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### If Jupyter doesn't start
```powershell
pip install jupyter notebook --force-reinstall
```

### If GPU not detected
```python
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
```

### Clear cache
```powershell
pip cache purge
```

## Useful Links

- **TensorFlow GPU Guide**: https://www.tensorflow.org/install/gpu
- **Pandas Documentation**: https://pandas.pydata.org/docs/
- **Scikit-learn**: https://scikit-learn.org/stable/
- **Matplotlib Gallery**: https://matplotlib.org/stable/gallery/
