# bigcodeModelsPlusCodeT
- here you find the code for implementing some LLM
- if you have Windows please make sure that you change in the *dynamic_module_utils.py* the 'signal.signal(...,...)' and the 'signal.alarm(..)' and 'signal.stop' as it only functions with Unix OS. For Windows replace this code with:
  'Import threading
defwatchdog():
print('Watchdogexpired.Exiting...')
os._exit(1)

alarm=threading.Timer(TIME_OUT_REMOTE_CODE,watchdog)
alarm.start()
.
.
alarm.cancel()
- make sure you installed:
- pip install transformers
- pip install 'transformers[torch]' => cpu
- pip install 'transformers[tf-cpu]'
- pip install 'transformers[flax]'

For Anaconda / miniconda:

- conda install -c huggingface transformers
- conda install accelerate
