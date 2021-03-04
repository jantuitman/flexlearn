import re
import logging

class Manager:
  def __init__(self, tracker = None, verbose : bool = False):
    self.experiments = {}
    # tuples of (pattern, callback)
    self.events = {
      'setup_model' : [],
      'request_batch': [],
      'load_checkpoint': [],
      'save_checkpoint': [],
      'train' : [],
      'validate': [],
      'check_validation_result': []
    }
    self.logger = logging.getLogger('flexlearn')
    self.tracker =tracker
    if verbose:
      print("verbose logging")
      ch = logging.StreamHandler()
      ch.setLevel(logging.DEBUG)

      # create formatter
      formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

      # add formatter to ch
      ch.setFormatter(formatter)

      self.logger.addHandler(ch)
    else:
      ch = logging.NullHandler()
      self.logger.addHandler(ch)  

  #
  #
  #
  def add_experiment(self, name : str, params : dict):
    """
      @param name  - name of the experiment
      @param params - contains
                        max_epochs - max number of epochs
                        validation_after - after how many training epochs the validation is run
                        model_params - parameters voor the model 
                        training_set - the name of the set used to train 
                        validation_set - the name of the set used to evaluate


    """ 
    params['experiment_name'] = name
    self.experiments[name] = params
  

  

  def on_setup_model(self, experiment_pattern : str, callback ):
    self.store_event('setup_model', (experiment_pattern,callback))

  def on_load_checkpoint(self, experiment_pattern, callback):
    self.store_event('load_checkpoint', (experiment_pattern,callback))

  def on_save_checkpoint(self, experiment_pattern, callback):
    self.store_event('save_checkpoint', (experiment_pattern,callback))

  def on_request_batch(self, dataset_pattern, callback):  
    self.store_event('request_batch', (dataset_pattern,callback))

  def on_train(self, experiment_pattern, callback):
    self.store_event('train', (experiment_pattern,callback))
  
  def on_validate(self, experiment_pattern, callback):
    self.store_event('validate', (experiment_pattern,callback))


  def on_check_validation_result(self, experiment_pattern, callback):
    self.store_event('check_validation_result', (experiment_pattern,callback))

  def learn(self, experiment_name : str):
    if experiment_name in self.experiments:
      self.apply_learning(self.experiments[experiment_name])
      return True
    else:
      return False  


  ########
  #
  # implementation
  #
  ##########
  def apply_learning(self, experiment):
    model = self.apply_rule('setup_model',experiment['experiment_name'],experiment['model_params'])
    # check_point_path = f"./checkpoints/{experiment['experiment_name']}"
    # TODO load checkpoint metadata, to check how far we are.
    current_epoch = 0
    current_validation_loss = 1e98

    # restore previous state
    if self.tracker is not None:
      self.apply_rule('load_checkpoint',experiment['experiment_name'],model, self.statehandler)
    
    while current_epoch < experiment['max_epochs']:
      self.logger.info(f"----- EPOCH {current_epoch} ----------------------------------------------")
      #get dataset
      batch_iterator = self.apply_rule('request_batch',experiment['training_set'])
      batch_count = 0
      for batch in batch_iterator: 
        training_loss = self.apply_rule('train',experiment['experiment_name'], model, batch)
        self.logger.info(f"batch {batch_count} : loss {training_loss}")
        batch_count += 1


      if current_epoch !=0 and (current_epoch % experiment['validation_after'] == 0):   
        if self.tracker is not None:
          self.apply_rule('save_checkpoint',experiment['experiment_name'],model, self.statehandler)

        validation_batch_iterator = self.apply_rule('request_batch',experiment['validation_set'])  
        
        loss = 0
        for batch in validation_batch_iterator: 
          loss += self.apply_rule('validate',experiment['experiment_name'], model, batch)

        should_stop = self.apply_rule('check_validation_result',experiment['experiment_name'], loss)
        if should_stop:
          break

      current_epoch += 1    


  def apply_rule(self, phase : str, discriminator : str, *args):
    if phase in self.events:
      for pattern, callback in self.events[phase]:
        if re.match(pattern, discriminator):
          return callback(discriminator, *args)

  
  def store_event(self, event_name, event):
    self.events[event_name].append(event)