## Stratified Sampling:

![Stratified Sampling](./static/img/stratified_sampled.png)

[WallStreetMojo - Stratified Sampling | Example #.1 Source](https://www.wallstreetmojo.com/stratified-sampling/)

```python
from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(dataset, dataset["income"]):
    strat_train_set = dataset.loc[train_index]
    strat_test_set = dataset.loc[test_index]
```
[Code Credits from hands-on-ml2-notebooks , CH.2](https://machine-learning-apps.github.io/hands-on-ml2/02_end_to_end_machine_learning_project)

- `test_size` refers to the amount of the dataset you're sub-setting (here it's set to 20%)
- `random_state` is an optional parameter that's great to use as static seed, and replicate a certain pattern of randomness again. *(E.G: I got 2 datasets, data and labels, with the same size and indices, I can just both Shuffle them by the same seed, and they will match!)*

In the **for loop**, Iterating over the Stratified Split data specifically by the `income` parameter, so it will basically sample an income distribution proportional to our dataset. *(For the the `strat_train_set` Stratified Training Dataset, & `strat_test_set` Stratified Test Dataset)*

**Notice how both stratified datasets have similar proportions :D**
![Output of strat_"](./static/img/stratified_jupyter.png)


## Random Sampling:

![Random Sampling](./static/img/stratified_sampled.png)

