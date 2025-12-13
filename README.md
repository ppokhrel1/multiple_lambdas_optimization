<<<<<<< HEAD
# Generalized Multi-source Assimilation: A Framework for Cross-Modal Integration and Source Optimization

**Authors**: Pujan Pokhrel, Austin Schmidt, Elias Ioup, Mahdi Abdelguerfi

This repository contains code for optimizing multiple lambda values across various scenarios, focusing on applications such as data assimilation, expert routing, and large-scale optimization problems. The goal is to improve computational efficiency and accuracy in environmental, logistics, and related scientific tasks. 

The framework supports rigorous constraint satisfaction, scalable integration, and automatic tuning of source reliability. It is designed for large-scale scientific and logistical applications where combining information from heterogeneous modalities (e.g., sensor networks, numerical models, expert systems) is critical to accuracy and decision-making.

We propose a flexible and scalable architecture that performs:

- **Multi-source data assimilation**
- **Cross-modal optimization of lambda hyperparameters**
- **Physics-informed modeling using PDE solvers**
- **Expert system integration via Lagrangian-based routing**

---


## 📘 Theoretical Highlights

We establish theoretical foundations for learning optimal weights in multi-source systems through a **two-timescale augmented Lagrangian approach**. This formulation:

- **Automatically learns weights** over different numerical methods, ML models, or noisy measurements
- Ensures **numerical stability** and provides **convergence guarantees**:
  - Model parameters: **O(1/√k)**
  - Source weights: **O(1/k)**
- Derives **explicit bounds** for:
  - Numerical precision in automatic differentiation
  - Stability conditions under varying source reliability

Compared to Softmax-based routing, our Lagrangian method provides **hard constraint satisfaction**, interpretable weights, and improved robustness in noisy or dynamic settings.

---

## 🔬 Experiments

We evaluate our framework across **four core tasks**:

### 1. **Multi-Source Neural PDE Integration**
- Combines solvers: FNO, WENO-like networks, boundary-aware and multiscale solvers
- Demonstrates up to **18% MSE improvement** with Lagrangian vs Softmax
- Trained on Navier-Stokes equations with varying initial conditions and Gaussian noise

### 2. **Expert Routing in Physical Systems**
- Learns to select PDE solvers based on local flow regimes
- Lagrangian method offers sharper, regime-aware specialization
- Maintains **physical consistency (divergence < 10⁻⁴)** and reduces shock error

### 3. **Large-Scale Multi-Source Integration**
- 128 sources, each with their own transform network
- Lagrangian prioritizes accurate sources, while Softmax distributes more evenly
- Results highlight a tradeoff: **Softmax for robustness**, **Lagrangian for precision**

### 4. **Real-World Data Assimilation**
- Combines 5 measurement sources with varying noise, bias, and missing data
- Physics-constrained loss ensures adherence to Burgers' equation and mass conservation
- Lagrangian excels in **physics loss minimization**, even under noisy data

---

## 📁 File Structure

```
├── README.md
├── data_assimilation_test.py
├── expert_routing_test.py
├── large_scale_multi_source_test.py
├── multi_source_pde_solver_test.py
├── testing_data_assimilation.py
```

---

## 🧪 Getting Started

To clone the repo:

```bash
git clone git@github.com:ppokhrel1/multiple_lambdas_optimization.git
cd multiple_lambdas_optimization
```

To run a sample assimilation test:

```bash
python data_assimilation_test.py
```

To evaluate expert routing performance:

```bash
python expert_routing_test.py
```

Make sure to configure the input parameters and lambda values in the scripts or via config files.

---

## 🧠 Citation

If you find this work useful in your research, please cite:

```bibtex
@article{pokhrel2025generalized,
  title={Generalized Multi-source Assimilation: A Framework for Cross-Modal Integration and Source Optimization},
  author={Pokhrel, Pujan and Schmidt, Austin and Ioup, Elias and Abdelguerfi, Mahdi},
  journal={To appear},
  year={2025}
}
```

---

## 📬 Contact

For questions, feedback, or collaboration inquiries, reach out to [Pujan Pokhrel](mailto:pujan@pokhrel.org).
=======
# multiple_lambdas_optimization
Repository for multiple lambda optimization in different scenarios
>>>>>>> 0707c66 (Initial commit)
