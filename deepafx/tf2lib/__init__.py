import tensorflow as tf

from deepafx.tf2lib.data import *
from deepafx.tf2lib.image import *
from deepafx.tf2lib.ops import *
from deepafx.tf2lib.utils import *
from deepafx.tf2lib.specs import *

physical_devices = tf.config.experimental.list_physical_devices('GPU')
for d in physical_devices:
    tf.config.experimental.set_memory_growth(d, True)
