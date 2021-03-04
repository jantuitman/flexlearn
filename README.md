# FlexLearn

Flexlearn is a training manager to track machine/deep learning experiments, currently extremely alpha.
I started writing this because i needed a way to have different experiments trained from a Jupyter notebook with different datasets, and collect all the results.

Testable stuff in Jupyter notebooks often takes the form of a method e.g. ``def small_task(): .....`` . That way, you can stick the method in one Jupyter cell, and do a illustrative call to it in the next cell.

Flexlearn capitalizes on this functional style by allowing you to define small methods to produce your datasets, your models, and doing your learning.

## Example usage

This is a very basic example which you will also find in the tests.

```python
 from flexlearn import Manager

 log = []
  
  def setup_model(experiment_name, params):
    log.append('setup_model')
  
  def load_checkpoint(experiment_name, model, checkpoint_path):
    log.append('load_checkpoint')

  def save_checkpoint(experiment_name, model, checkpoint_path):
    log.append('save_checkpoint')  
  
  def request_batch(dataset_name):
    log.append('request_batch')
    if dataset_name == 'my_training_set':
      return iter([1 , 2, 3, 4])
    else:
      return iter([1 ])

  def train(experiment_name, model, batch):
    log.append(f'train {batch}')  
    return 1
  
  def validate(experiment_name, model, batch):
    log.append(f'validate {batch}')
    return 1

  def check_validation_result(experiment_name, loss):
    log.append('check_validation_result')
    return False  

  manager = Manager(verbose = True)
  manager.add_experiment("my_experiment", {
    'max_epochs': 4,
    'validation_after': 2,
    'training_set': 'my_training_set',
    'validation_set': 'my_validation_set',
    'model_params': {
      'dimension': 1000
    }
  })
  
  # the regexp defines which experiments/datasets must be handled by these functions
  # if you have 2 different functions for 2 datasets, you could implement
  # 2 versions of on_request_batch and register them with different regexps
  #
  # similarly if you have 2 experiments with different models,
  # you might have more setup_model or train/validate functions.
  manager.on_setup_model(r'.*', setup_model)
  manager.on_load_checkpoint(r'.*', load_checkpoint)
  manager.on_save_checkpoint(r'.*', save_checkpoint)
  manager.on_request_batch(r'.*', request_batch)
  manager.on_train(r'.*', train)
  manager.on_validate(r'.*', validate)
  manager.on_check_validation_result(r'.*', check_validation_result)
  
  # the manager does nor know other_experiment so it does nothing.
  assert False == manager.learn("other_experiment")

  # the manager does know my_experiment. it is currently not configured
  # for serializing so it won't call on_load_checkpoint, on_save_checkpoint.
  assert True == manager.learn("my_experiment")
  assert ['setup_model',
 'request_batch',
 'train 1',
 'train 2',
 'train 3',
 'train 4',
 'request_batch',
 'train 1',
 'train 2',
 'train 3',
 'train 4',
 'request_batch',
 'train 1',
 'train 2',
 'train 3',
 'train 4',
 'request_batch',
 'validate 1',
 'check_validation_result',
 'request_batch',
 'train 1',
 'train 2',
 'train 3',
 'train 4'] == log


```