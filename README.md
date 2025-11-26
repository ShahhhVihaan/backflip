# Backflip

3D Humanoid Backflip done the good old way with offline trajectory optimization and online MPC for tracking.

## Dev

```bash
conda create -n backflip python=3.11 pinocchio casadi -c conda-forge
conda activate backflip
python -c "from pinocchio import casadi as cpin; print('ok')"
pip install meshcat mujoco numpy scipy
```
