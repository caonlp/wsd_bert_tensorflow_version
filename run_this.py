import os
import tensorflow as tf
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


print(tf.__version__)

lst = os.listdir(os.getcwd())

for c in lst:
    if os.path.isfile(c) and c.endswith('.py') and c.find("run_this") == -1:
        # print(c)
        os.system('python {}'.format(c))