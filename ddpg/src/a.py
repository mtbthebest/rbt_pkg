#!/usr/bin/env python

import tensorflow as tf
import numpy as np
import tensorflow as tf
import pickle

# output = open('/home/mtb/rbt_ws/src/ddpg/src/data.txt', 'wb')
# b = 5
# for i in range(4):
#     b +=5
#     c=[i,b]
#     pickle.dump(c, output)

# output.close()
output = open('/home/mtb/rbt_ws/src/ddpg/scripts/rewards.txt', 'rb')
while True:
    try:
        print pickle.load(output)
    except:
       break
output.close()