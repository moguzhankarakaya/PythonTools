import os, inspect, mmap, sys, shutil

_MODE_SEPERATOR     = [';', '->',]  # Config file MODE seperator
_TIME_OUT           = 10*60         # 10 mins
_RD_PWD             = 'pwd'         # Remote Connection Password
_MODE               = None          # Global Run Mode
_CONTEXT            = None          # Global Run Context
_CALLER             = None          # Global Initilizer Script
_REQUIRES_CLEANUP   = False
_ROOT_APP_PATH      = os.getcwd()

def callerPath():
  frame  = inspect.stack()[1]
  module = inspect.getmodule(frame[0])
  return os.path.dirname(module.__file__)

def getRunMode(caller):
  configFile = os.path.join(caller, 'application.conf')
  mode, context = None, None
  with open(configFile, "r") as config:
    for line in config:
      if "MODE" in line:
        runSpecs = line.split("=")[1]
        if any([spr in runSpecs for spr in _MODE_SEPERATOR]):
          for sep in _MODE_SEPERATOR[1:]:
            runSpecs = runSpecs.replace(sep, _MODE_SEPERATOR[0])
            mode, context = runSpecs.split(_MODE_SEPERATOR[0])
            mode = mode.strip()
            context = context.strip()
        else:
          mode = runSpecs.strip()
  if not mode:
    mode = "RELEASE"
  else:
    if mode=="DEBUG":
      if not context:
        context = "ALL"
  return mode, context

def getContextList(caller, rawContext):
  contextList = []
  contextName = rawContext.split(',')
  for context in contextName:
    contextList.append(os.path.join(caller, context.strip(),'main.py'))
  return contextList

def setDebugEnvironment(caller):
  mode, rawContext = getRunMode(caller)
  global _MODE; global _CONTEXT; global _CALLER; global _REQUIRES_CLEANUP
  _MODE, _CONTEXT, _CALLER = mode, rawContext, caller
  if _MODE == 'DEBUG':
    if not breakPointExist():
      if _CONTEXT == 'ALL':
        # Inject break point in main file under src folder
        _CONTEXT = getContextList(caller, 'src')
      else:
        # Inject break point in context
        _CONTEXT = getContextList(caller, _CONTEXT)
      injectBreakPoint(_CONTEXT)
      _REQUIRES_CLEANUP = True
  os.environ['RD_RUN_MODE'] = _MODE

def cleanupEnvironment():
  global _MODE; global _CONTEXT; global _REQUIRES_CLEANUP
  if _MODE == 'DEBUG':
    if _REQUIRES_CLEANUP:
      retriveBackups(_CONTEXT)
  os.environ.pop('RD_RUN_MODE')

def injectBreakPoint(scripts):
  for script in scripts:
    injectionDone = False
    shutil.copy(script, script + '.bak')
    with open(script, 'r') as inFile:
      with open(script + '.mod', 'w') as outFile:
        for line in inFile:
          if not injectionDone:
            if 'def main():' in line:
              outFile.write(line)
              outFile.write('  from Debugger import setBreakPoint\n  setBreakPoint()\n')
              injectionDone = True
              continue
          outFile.write(line)
    shutil.copy(script + '.mod', script)
    os.remove(script + '.mod')

def retriveBackups(scripts):
  for script in scripts:
    shutil.copy(script + '.bak', script)
    os.remove(script + '.bak')

def scanBreakPoints():
  scripts = []
  breakPoints = []
  for dirpath, _, filenames in os.walk(_ROOT_APP_PATH):
    for filename in [f for f in filenames if f.endswith(".py")]:
      if 'Debugger.py' == filename:
        continue
      scripts.append(os.path.join(dirpath, filename))
  for script in scripts:
    if os.stat(script).st_size == 0:
      continue
    with open(script, 'rb', 0) as file:
      with mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ) as s:
        if s.find(br'setBreakPoint()') != -1:
          breakPoints.append(script)
  return breakPoints

def breakPointExist():
  listOfBreakPoints = scanBreakPoints()
  if len(listOfBreakPoints) > 0:
    return True
  return False

def setBreakPoint():
  global _RD_PWD; global _TIME_OUT
  if os.environ['RD_RUN_MODE'] == 'DEBUG':
    import rpdb2; rpdb2.start_embedded_debugger(_rpdb2_pwd   =_RD_PWD, 
                                                fAllowRemote =True, 
                                                timeout      =_TIME_OUT)
