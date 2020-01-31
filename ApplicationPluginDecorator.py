import functools, time, weakref, sys, os, logging

class ApplicationPluginDecorator(object):
  '''
    Application/Callback wrapper
    A decorator class which takes callable input function and 
    decorates it with pre&post events. Attributes are settable. 
    It tracks its own instances and if used in context manager 
    proper clean up can be done.
  '''
  _instances = set()
  
  def __init__(self, func):
    '''
      Constructor takes a callable to decorate
    '''
    setupLogger('log_dump.txt')
    self._instances.add(weakref.ref(self))
    if callable(func):
      functools.update_wrapper(self, func)
      self.func = func
    else:
      _raiseTypeError(func, 'func', 'callable')
    self.preCallables  = []
    self.postCallables = []
    self.abortFunc     = None
    self.cleanUpFunc   = None
    self.app           = None
    self.coolDown      = 1.0
    self.isAppRunning  = False

  @classmethod
  def getinstances(self):
    '''
      Returns its instances.
      Use this under context manager to ensure all created 
      instances are destroyed with proper post function calls
    '''
    dead = set()
    for ref in self._instances:
        obj = ref()
        if obj is not None:
            yield obj
        else:
            dead.add(ref)
    self._instances -= dead


  @property
  def preCallables(self):
    '''
      Returns Pre-Callable types.
    '''
    return self.__preCallables

  @preCallables.setter
  def preCallables(self, preCallables):
    '''
      Sets Pre-Callable types list.
      List should contain tuples where the first item is either before_app 
      or after_app that determines if the callable will be called before or 
      after target app is started.
    '''
    self.__preCallables  = _checkCallables(preCallables,  'preCallables' )

  @property
  def postCallables(self):
    '''
      Returns Post-Callable types.
    '''
    return self.__postCallables


  @postCallables.setter
  def postCallables(self, postCallables):
    '''
      Sets Post-Callable types list.
      List should contain tuples where the first item is either before_app 
      or after_app that determines if the callable will be called before or 
      after target app is stopped.
    '''
    self.__postCallables  = _checkCallables(postCallables,  'postCallables' )

  @property
  def abortFunc(self):
    '''
      Get optional abort function that will be called if starting application fails.
    '''
    return self.__abortFunc


  @abortFunc.setter
  def abortFunc(self, abortFunc):
    '''
      Set optional abort function that will be called if starting application fails.
    '''
    self.__abortFunc  = abortFunc if callable(abortFunc) and abortFunc.__code__.co_argcount == 0 else None

  @property
  def cleanUpFunc(self):
    '''
      Get optional clean-up function that will be called during clean-up of existing instances.
    '''
    return self.__cleanUpFunc

  @cleanUpFunc.setter
  def cleanUpFunc(self, cleanUpFunc):
    '''
      Set optional clean-up function that will be called during clean-up of existing instances.
    '''
    self.__cleanUpFunc  = cleanUpFunc if callable(cleanUpFunc) and cleanUpFunc.__code__.co_argcount == 0 else None

  @property
  def app(self):
    '''
      Get application context, as returned from provided application start function
    '''
    return self.__app

  @app.setter
  def app(self, app):
    '''
      Set application context, as returned from provided application start function
    '''
    self.__app = app
  
  @property
  def coolDown(self):
    '''
      Get cooldown time that will be used to wait after application is stopped.
    '''
    return self.__coolDown

  @coolDown.setter
  def coolDown(self, period):
    '''
      Set cooldown time that will be used to wait after application is stopped.
    '''
    if isinstance(period, (int, float)):
      self.__coolDown = period
    else:
      raise PluginException(TypeError,
              "Passed argument with type '{}' is not an integer object".format(type(period))
            )

  def _postCalls(self, pos):
    '''
      Private function that invokes provided Post-Callables
    '''
    for postCallableTuple in self.postCallables:
      if postCallableTuple[0] == pos:
        if len(postCallableTuple) > 2:
          try:
            postCallableTuple[1](postCallableTuple[2:])
          except Exception as err:
            logWarning("Error happened in {} while calling {}: \n\tException: {}".format(localFunctionName(), postCallableTuple[1].__name__, str(err)))
        else:
          postCallableTuple[1]()

  def _preCalls(self, pos):
    '''
      Private function that invokes provided Pre-Callables
    '''
    for preCallableTuple in self.preCallables:
      if preCallableTuple[0] == pos:
        if len(preCallableTuple) > 2:
          try:
            preCallableTuple[1](preCallableTuple[2:])
          except Exception as err:
            logWarning("Error happened in {} while calling {}: \n\tException: {}".format(localFunctionName(), preCallableTuple[1].__name__, str(err)))
        else:
          preCallableTuple[1]()
  
  def stopProcess(self):
    '''
      Stopping process of application.
      This routine calls post callables before 
      and after stopping the application.
    '''
    if self.app:
      self._postCalls('before_app')
      
      try:
        self.app.detach()
        self.app = None
        time.sleep(self.coolDown)
        logMsg("Application has been stopped successfully.")
        self.isAppRunning = False
      except Exception as err:
        logWarning("Application couldn't be stopped properly. Exception: {}".format(str(err)))
        if callable(self.cleanUpFunc):
          self.cleanUpFunc()

      if self.app == None:
        self._postCalls('after_app')
    
    self.app = None
    
  def __call__(self, *args, **kwargs):
    '''
      Overload of instance call function. Original application call functions
      decorates it in a way that it calls pre-callables before and after 
      calling apllication callable.
    '''
    if self.isAppRunning:
      self.stopProcess()
    self._preCalls('before_app')
    
    try:
      if self.argList:
        args = self._concatArguments(args)
      self.app = self.func(*args, **kwargs)
      self.isAppRunning = self.app is not None
    except Exception as err:
      self.app = None
      logWarning("Application has failed to start. Exception: {}".format(str(err)))
      if callable(self.abortFunc):
        self.abortFunc()
    
    if self.app:
      self._preCalls('after_app') 
    
    return self.app
  
  def __del__(self):
    '''
      Destructor overload that stops the application and calls cleanup function(s).
    '''
    logMsg("Application Plugin is stopped safely")
    self.stopProcess()
    if callable(self.cleanUpFunc):
      self.cleanUpFunc()


class PluginException(Exception):
  def __init__(self,*args,**kwargs):
    Exception.__init__(self,*args,**kwargs)

def _raiseTypeError(obj, param, targetType):
  raise PluginException(TypeError,
          "Passed argument with type '{}' to initialize {} is not {} object".format(type(obj),param, targetType)
        )

def _checkCallables(objs, init):
  if isinstance(objs, list):
    for objTuple in objs:
      if not callable(objTuple[1]):
        _raiseTypeError(objTuple[1], init, 'callable')
      if not (objTuple[0] == 'before_app' or objTuple[0] == 'after_app'):
        raise PluginException(KeyError,
                "For '{}' as '{}', wrong keyword has been specified: '{}'.".format(objTuple[1].__name__, init, objTuple[0]) + \
                " It must be either 'before' or 'after'."
              )
    return objs
  return []

def localFunctionName(frame=1):
  caller = sys._getframe(frame)
  function = caller.f_code.co_name
  file = caller.f_code.co_filename
  module = os.path.splitext( os.path.basename(file) )[0]
  return module + '::' + function

def setupLogger(fileName, ignore=[]):
    '''
        Sets up logger with a given file name. 
        Add those modules that needs to be suppressed from logger output into ignore list.
    '''
    logging.basicConfig(filename=fileName,level=logging.DEBUG)
    for module in ignore:
        logging.getLogger(module).setLevel(logging.ERROR)

def logMsg(msg, std=False):
    '''
        Log info
    '''
    logging.info(msg)
    if std: print(msg)

def logWarning(msg, std=False):
    '''
        Log warning
    '''
    logging.warning(msg)
    if std: print(msg)  

def logError(msg, std=False):
    '''
        Log error
    '''
    logging.error(msg)
    if std: print(msg)