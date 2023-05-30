# Thesis-Code
This repo contains the code to simulate Proximity Operations

## Contents
- `Matlab/`: implementation of the Proximity Operation between Target and Chaser in matlab 
  - 'Tester.m': file needed to test the simulations and the dynamics 
  - '\*.m': various simulations files
- `Python/`: implementation of the Proximity Operation environments in Python 
  - 'Test_env': implementation of environments 
- `README.md`: this file.

## Usage
The python section of this git has a submodule with all the savings from each learning run. To use them, you need to clone the repo with the following command:
```
git clone --recurse-submodules $(ThisRepo)
```