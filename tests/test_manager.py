from pprint import pprint
from flexlearn import Manager

def test_learn():
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
  manager.on_setup_model(r'.*', setup_model)
  manager.on_load_checkpoint(r'.*', load_checkpoint)
  manager.on_save_checkpoint(r'.*', save_checkpoint)
  manager.on_request_batch(r'.*', request_batch)
  manager.on_train(r'.*', train)
  manager.on_validate(r'.*', validate)
  manager.on_check_validation_result(r'.*', check_validation_result)
  assert False == manager.learn("other_experiment")
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

def test_resume_learn_after_save():
  pass