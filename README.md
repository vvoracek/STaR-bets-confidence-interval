# STaR-Bets: Sequential Testing and Randomized Betting

A code for STaR-Bets paper. The `star()` function takes an i.i.d. sample supported on $[0,1]$ and returns a lower bound on the mean at confidence level $1-\alpha$.

## Usage

```python
from star import star
import numpy as np

data = np.random.rand(100)  # Your data here
lower_bound = star(data, alpha=0.05)    # 95% confidence lower bound
```

## Experiments
To replicate experiments in the paper, the core.py implements all the methods and experiment templates, and main.py runs the experiments under different settings.
