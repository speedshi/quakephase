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
Shi, P., Meier, M.-A., Villiger, L., Tuinstra, K., Selvadurai, P. A., Lanza, F., Yuan, S., Obermann, A., Mesimeri, M., Münchmeyer, J., Bianchi, P., and Wiemer, S. (2024). From labquakes to megathrusts: Scaling deep learning based pickers over 15 orders of magnitude. Journal of Geophysical Research: Machine Learning and Computation, 1(4), e2024JH000220. [https://doi.org/10.1029/2024JH000220](https://doi.org/10.1029/2024JH000220)

BibTex:  
```
@article{shi2024labquakes,
  title={{From labquakes to megathrusts: Scaling deep learning based pickers over 15 orders of magnitude}},
  author={Shi, Peidong and Meier, Men-Andrin and Villiger, Linus and Tuinstra, Katinka and Selvadurai, Paul Antony and Lanza, Federica and Yuan, Sanyi and Obermann, Anne and Mesimeri, Maria and M{\"u}nchmeyer, Jannes and Bianchi, Patrick and Wiemer, Stefan},
  journal={Journal of Geophysical Research: Machine Learning and Computation},
  volume={1},
  number={4},
  pages={e2024JH000220},
  year={2024},
  doi={10.1029/2024JH000220},
  publisher={Wiley Online Library}
}
```
