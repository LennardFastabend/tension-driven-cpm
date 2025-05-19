# Tension-Driven CPM

## 1. Introduction  
This repository contains code for the automated setup, execution, and analysis of Cellular Potts Model (CPM) simulations using CompuCell3D (CC3D). Growth is driven by local tissue tension, resulting in curvature-dependent proliferation under geometric confinement.

Example simulations and templates for various substrate geometries are provided.

---

## 2. System Requirements

- **CompuCell3D (CC3D):** Version 4.3.2 or higher  
- **Python:** Version 3.11  
- **Required Python packages:**  
  - `pathlib`  
  - `numpy`  
  - `matplotlib`  
  - `opencv-python`

---

## 3. Repository Structure

- **`scenario/`**  
  Contains different simulation scenarios, each including at least one example simulation runnable with CC3D. Based on these example simulations, templates are provided for setting up parameter scans.

- **`parameter_scan/`**  
  Contains code to set up and execute parameter scans. Simulation outputs are stored in their respective simulation directories.

- **`data_analysis/`**  
  Contains scripts for analyzing simulation data.

---

## 4. Custom Simulations

To create custom simulations, it is recommended to reuse one of the provided examples and modify simulation parameters as needed. Custom substrate geometries can be generated similarly to the code in `piff_generator.py`.

**Important:**  
When using a custom substrate geometry, replace the `.piff` file in the simulation directory accordingly, and ensure the initial conditions in the main simulation file refer to the new `.piff` file.

---

## 5. Monolayer Tutorial

This section demonstrates how to perform and analyze a parameter scan using the **monolayer** scan scenario. The structure presented here can be adapted to other scenarios.

### Parameter Scan: Setup and Execution  
Run the script `MainMonolayerParameterScan` to set up and execute an example parameter scan of the monolayer scenario.

### Data Analysis  
Run the script `MainMonolayerAnalysis` to analyze the results of the parameter scan.

---