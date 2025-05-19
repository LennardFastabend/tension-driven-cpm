# Curved Tissue CPM

## 1. Introduction
This repository includes code for the automated setup, execution, and analysis of CPM (Cellular Potts Model) simulations of curved tissue development using CC3D. Growth is driven by local tissue tension, resulting in curvature-dependent proliferation under geometric confinement.

Example simulations and templates for various substrate geometries are provided.

## 2. System Requirements

- **CC3D**: Version 4.3.2 or higher *(specify exact version)*
- **Python**: 3.11
  Required packages:
  - pathlib
  - numpy
  - matplotlib
  - opencv-python

## 3. Repository Structure
Different simulation scenarios are given in the "scenario" directory with at least one example simulation, that can be executed using CC3D. Based on this example simulation a template is given to set up parameter scans. 

To set up and run parameter scans code is provided in the "parameter_scan" directory. The simulation output is generated in the respective directories of the individual simulations. Code for data analysis is given in the "data_analysis" directory. A simple example to run and analyse an exemplary parameter scan of a monolayer simulation is described in the following section.

To set up custom simulations it is advised to reuse one of the given examples and tweak simulation parameters as needed. Custom Substrate geometries can be generated similar to the code provides in "piff_generator.py". Note that the piff-file has to be exchanged in the simulation directory and the setup of the initial conditions has to refere to the new piff-file in the main simulation file.

## 4. Monolayer Tutorial
This section provides an example of how to perform and analyse a parameter scan using the ‘monolayer’ scan scenario. The structure of the code given here can be transferred to scans of other scenarios.

### Parameter Scan: Setup and Execution
To set up an exemplary parameter scan of the monolayer scenario run the file "MainMonolayerParameterScan".

### Data Analysis
To set up an exemplary parameter scan of the monolayer scenario run the file "MainMonolayerAnalysis".

# To Do: only publish needed code files! -> Sort