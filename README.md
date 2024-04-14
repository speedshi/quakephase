<p align="center">
  <img src="https://github.com/speedshi/quakephase/blob/main/docs/figs/QUAKEPHASE_logo.png" />
</p>

---

# quakephase
Machine-learning based toolbox to characterize seismic phases, i.e. phase detection, phase classification, and phase picking.  
Currently, the **quakephase** toolbox can be used to largely enhance the pre-trained ML seismic phase picking models.  


## Installation
Install using pip (recommend)
```bash
pip install quakephase
```

Install from source code
```bash
conda create -n quakephase python=3.9
conda activate quakephase
git clone https://github.com/speedshi/quakephase.git
cd quakephase
pip install .
```


## Usage
Follow the example scripts to use **quakephase**:  
**use_quakephase_example.py**  
**use_quakephase_example.ipynb**  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/speedshi/quakephase/blob/main/use_quakephase_example.ipynb)  
  
Input parameters are explained and set in the parameter YAML file: "parameters.yaml". Feel free to explore and play with different parameters.  
Note for data with distinct data sampling rates, the rescaling factor affects the performance most, choose it wisely. And if computing power is allowed, you can assemble different sets of rescaling rates to maximize the performance.  


## Reference 
Please cite the following paper in your documents if you use **quakephase** in your work:  
Peidong Shi, Men-Andrin Meier, Linus Villiger, Katinka Tuinstra, Paul Selvadural, Federica Lanza, Sanyi Yuan, Anne Obermann, Maria Mesimeri, Jannes Münchmeyer, Patrick Bianchi, and Stefan Wiemer. From labquakes to megathrusts: Scaling deep learning based pickers over 15 orders of magnitude. ESS Open Archive. April 12, 2024. DOI: [10.22541/essoar.171291855.54784565/v1](https://doi.org/10.22541/essoar.171291855.54784565/v1)


