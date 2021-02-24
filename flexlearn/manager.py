class Manager:
  def __init__(self):
    self.experiments = {}
    self.datasets = {}
    self.events = {
      'setup_model' : [],
      'load_checkpoint': [],
      'train' : [],
      'validate': [],
      'check_validation_result': []
    }

  def add_dataset(self, name : str, batch_iterator : iter):
    pass
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
    self.experiments[name] = params
  
  

  def on_setup_model(self, experiment_pattern : str, callback ):
    pass

  def on_load_checkpoint(self, experiment_pattern, callback):
    pass

  def on_train(self, experiment_pattern, callback):
    pass
  
  def on_validate(self, experiment_pattern, callback):
    pass
  
  def on_check_validation_result(self, experiment_pattern, callback):
    pass

  def learn(self, experiment_name : str):
    if experiment_name in self.experiments:
      return True
    else:
      return False  
