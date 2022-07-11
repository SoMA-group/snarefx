#*************************************************************************
#  Copyright 2021 Adobe Systems Incorporated.
#
#  Please see the attached LICENSE file for more information.
#
#**************************************************************************/
import numpy as np

k = {}

k['sr'] = 44100
k['num_samples'] = 22050
k['batch_size'] = 100
# k['steps_per_epoch'] = 1000
k['steps_per_epoch'] = int( np.ceil(3000/ k['batch_size']) )

k['epochs'] = 1000
k['patience'] = 25
k['encoder'] = 'inception' # inception or mobilenet

# Define paths
k['path_audio'] = '/home/code-base/scratch_space/top2bottom_noNorm/'
k['path_models'] = '/home/code-base/scratch_space/models/top2bottomModels/'


# DAFX constants
k['output_length'] = 1024 # Length of output frame
k['hop_samples'] = 1024 # Length of hop size, for non-overlapping frames it should be equal to output_length
k['gradient_method'] = 'spsa' # or 'spsa'
k['compute_signal_gradient'] = False
k['multiprocess'] = True
k['greedy_dafx_pretraining'] = False # Enables progressive training
k['default_pretraining'] = False # Enables default initialization of parameter values for training

k['params'] = []
k['plugin_uri'] = []
k['param_map'] = []
k['stereo'] = []
k['set_nontrainable_parameters'] = []
k['new_parameter_range'] = []


# Single DAFx:
#k['params'] = []
#k['plugin_uri'] = []
#k['param_map'] = []
#k['stereo'] = []
#k['set_nontrainable_parameters'] = []
#k['new_parameter_range'] = []
#
# # Multiband compressor
#
#k['plugin_uri'] = 'http://calf.sourceforge.net/plugins/MultibandCompressor'
#k['stereo'] = True
# # params = [19,20,30,31,41,42,52,53,15,16,17] # threshold, ratio, freqs.
# # params = [19,20,23,30,31,34,41,42,45,52,53,56,15,16,17] # threshold, ratio, makeupgain, freqs.
# # params = [19,20,23,30,31,34,41,42,45,52,53,56,15,16,17,5] # threshold, ratio, makeupgain, freqs, out gain
# # params = [19,20,23,30,31,34,41,42,45,52,53,56,15,16,17,5,6] # threshold, ratio, makeupgain, freqs, input output gain gain
#
#params = [19,20,23,24,30,31,34,35,41,42,45,46,52,53,56,57,15,16,17,5,6] # threshold, ratio, makeupgain, knee, freqs, input output  gain
#
#k['set_nontrainable_parameters'] = {28:0,
#                                     39:0,
#                                     50:0,
#                                     61:0,
#                                     21:0.01,
#                                     22:0.01,
#                                     32:0.01,
#                                     33:0.01,
#                                     43:0.01,
#                                     44:0.01,
#                                     54:0.01,
#                                     55:0.01,
#k['params'] = []
#k['plugin_uri'] = []
#k['param_map'] = []
#k['stereo'] = []
#k['set_nontrainable_parameters'] = []
#k['new_parameter_range'] = []#                                    }
#
#k['new_parameter_range'] = {15:[10.0, 300.0],
#                             16:[300.0, 3000.0],
#                            17:[3000.0,8000.0],
#                            5:[0.015625,2],
#                            6:[0.015625,2]
#                            }
#
#k['params'] = len(params)
#k['param_map'] = {}
#for i, port in enumerate(params):
#	k['param_map'][i] = port
#
# MULTIPLE DAFx:


### # # Multiband Compressor
#
#plugin_uri = 'http://calf.sourceforge.net/plugins/MultibandCompressor'
#stereo = True
##parameters = [19,20,23,30,31,34,41,42,45,52,53,56,15,16,17,5,21,22,32,33,43,44,54,55] # threshold, ratio, makeupgain, freqs, input gain, 
#parameters = [19,20,23,30,31,34,41,42,45,52,53,56,15,16,17,5] # threshold, ratio, makeupgain, freqs, input gain, 
#
#
#set_nontrainable_parameters = {28:0,
#                                    39:0,
#                                    50:0,
#                                    61:0,
#                                    21:0.01, #Attack 1
#                                    22:0.01, #Release 1
#                                    32:0.01, #Attack 2
#                                    33:0.01, #Release 2
#                                    43:0.01, #Attack 3
#                                    44:0.01, #Release 3
#                                    54:0.01, #Attack 4
#                                    55:0.01, #Release 4
#                                   }
#
#new_parameter_range = { 15:[10.0, 400.0],  #Split 1/2
#                       16:[400.0, 3000.0], #Split 2/3
#                       17:[3000.0,10000.0], #Split 3/4
#                       23:[1.0, 10.0], #makeupgain1
#                       34:[1.0, 10.0], #makeupgain2
#                       45:[1.0, 10.0], #makeupgain3
#                       56:[1.0, 10.0], #makeupgain4
##                        21:[0.010000, 1000.0],
##                        22:[0.010000, 1000.0],
##                        32:[0.010000, 1000.0],
##                        33:[0.010000, 1000.0],
##                        43:[0.010000, 1000.0],
##                        44:[0.010000, 1000.0],
##                        54:[0.010000, 1000.0],
##                        55:[0.010000, 1000.0],
#                           5:[0.015625,4], #level in gain
##                            6:[0.015625,5]
#                           }
##
#params = len(parameters)
#param_map = {}
#for i, port in enumerate(parameters):
#    param_map[i] = port
#k['params'].append(params)
#k['plugin_uri'].append(plugin_uri)
#k['param_map'].append(param_map)
#k['stereo'].append(stereo)
#k['set_nontrainable_parameters'].append(set_nontrainable_parameters)
#k['new_parameter_range'].append(new_parameter_range)

## # Parametric EQ
#
##
#plugin_uri = 'http://calf.sourceforge.net/plugins/Equalizer8Band'
#stereo = True
#parameters = [22,23,25,26,28,29,30,32,33,34,36,37,38,40,41,42,5,6] # freq, gain, Q * 4 bands, freq, gain * 4 bands, input gain, output gain
#
#
#set_nontrainable_parameters = {15:0, # HP
#                                18:0, # LP
#                                21:1, # LS
#                                24:1, # HS
#                                27:1, # F1
#                                31:1, # F2
#                                35:1, # F3
#                                39:1, # F4
#                                16:30,
#			16:10.0, #hp_freq
#                        19:20000, #lp_freq
##                        22:1, #ls_level
##                        23:100, #ls_freq
##                        25:1, #hs_level
##                        26:5000, #hs_freq
#                        #28:1,
#                        #29:200,
#                        #30:1,
#                        #32:1,
#                        #33:500,
#                        #34:1,
#                        #36:1,
#                        #37:2000,
#                        #38:1,
#                        #40:1,
#                        #41:4000,
#                        #42:1,
#                          }
#
#new_parameter_range = { #16:[10.0, 100.0], #hp_freq
#                        22:[0.015625, 5], #ls_level
#                        23:[10.0, 1000.0], #ls_freq
#                        25:[0.015625, 5], #hs_level
#                        26:[5000.0, 20000.0], #hs_freq
#                        28:[0.015625, 5], #p1_level
#                        29:[10.0, 1000.0], #p1_freq
#                        30:[0.1, 20.0], #p1_q
#                        32:[0.015625, 5], #p2_level
#                        33:[400.0, 1100.0], #p2_freq
#                        34:[0.1, 20.0], #p2_q
#                        36:[0.015625, 5], #p3_level
#                        37:[900.0, 4000.0], #p3_freq
#                        38:[0.1, 20.0], #p3_q
#                        40:[0.015625, 5], #p4_level
#                        41:[1000.0, 20000.0], #p4_freq
#                        42:[0.1, 20.0], #p4_q
#                        5:[0.015625, 2.0], #level_in
#                        6:[0.015625, 2.0] #level_out
#                            }
#
#params = len(parameters)
#param_map = {}
#for i, port in enumerate(parameters):
#     param_map[i] = port
#    
#k['params'].append(params)
#k['plugin_uri'].append(plugin_uri)
#k['param_map'].append(param_map)
#k['stereo'].append(stereo)
#k['set_nontrainable_parameters'].append(set_nontrainable_parameters)
#k['new_parameter_range'].append(new_parameter_range)
##
## # Limiter-CALF
#
#
#
## plugin_uri = 'http://calf.sourceforge.net/plugins/Limiter'
## stereo = True
#
## parameters = [15, 17, 21, 5, 6] # limit, release, ASC coeff, input gain, output gain,
#
## # parameters = [15]
## set_nontrainable_parameters = {16:5.0}
#                                  
## new_parameter_range = {
##                         17:[1.,50.0],
##                         5:[0.015625, 2.0],
##                         6:[0.015625, 2.0]
##                         }
#
## params = len(parameters)
## param_map = {}
## for i, port in enumerate(parameters):
##     param_map[i] = port
#    
## k['params'].append(params)
## k['plugin_uri'].append(plugin_uri)
## k['param_map'].append(param_map)
## k['stereo'].append(stereo)
## k['set_nontrainable_parameters'].append(set_nontrainable_parameters)
## k['new_parameter_range'].append(new_parameter_range)
#
# # 32 Band Graph EQ
###
#plugin_uri = 'http://lsp-plug.in/plugins/lv2/graph_equalizer_x32_mono'
#stereo = False
#
#parameters = [18]
#for i in range(31):
#    parameters.append(parameters[i]+5)
#
#new_parameter_range = {}
#for i in parameters:
#    new_parameter_range[i] = [0.015850, 60.0]
#
## parameters.append(3)
#parameters.append(4)
#
## k['new_parameter_range'][3] = [0.0, 4.0]
#new_parameter_range[4] = [0.0, 4.0]
#
#set_nontrainable_parameters = {6:4}
#
#params = len(parameters)
#param_map = {}
#for i, port in enumerate(parameters):
#    param_map[i] = port
#
#k['params'].append(params)
#k['plugin_uri'].append(plugin_uri)
#k['param_map'].append(param_map)
#k['stereo'].append(stereo)
#k['set_nontrainable_parameters'].append(set_nontrainable_parameters)
#k['new_parameter_range'].append(new_parameter_range)
###

#
## Limiter -lsp
##
#plugin_uri = 'http://lsp-plug.in/plugins/lv2/limiter_mono'
#stereo = False
#
## parameters = [7,8,10,11,12] # th, knee, lookahead, attack, release
## parameters = [7,10] # th, lookahead,
#parameters = [7] # th,
## 
## parameters = [15]
#set_nontrainable_parameters = {}
#                                  
#new_parameter_range = {}
#
#params = len(parameters)
#param_map = {}
#for i, port in enumerate(parameters):
#    param_map[i] = port
#
#    
#k['params'].append(params)
#k['plugin_uri'].append(plugin_uri)
#k['param_map'].append(param_map)
#k['stereo'].append(stereo)
#k['set_nontrainable_parameters'].append(set_nontrainable_parameters)
#k['new_parameter_range'].append(new_parameter_range)
##
####Transient Designer 
#plugin_uri = 'http://calf.sourceforge.net/plugins/TransientDesigner'
#stereo = True
#parameters = [16, 17, 18, 19, 20] #attack time, attack boost,
####					sustain threshold, release time, release boost, lookahead
#set_nontrainable_parameters = { #5:1, #input gain
#				#6:1 #output gain
#			        } 
#
#new_parameter_range = { #5:[0.015625, 5], #in gain
#                        #6:[0.015625, 5], #out gain
#                        16:[1.0, 250.0], #attack time
#                        17:[-0.5, 0.5], #attack boost
#                        18:[0.0001, 1.0], #sustain threshold
#                        19:[1.0, 250.0], #release time
#                        20:[-0.5, 0.5], #release boost
#                        #23:[0, 10.0], #lookahead
#			}
#
#params = len(parameters)
#param_map = {}
#for i, port in enumerate(parameters):
#    param_map[i] = port
#k['params'].append(params)
#k['plugin_uri'].append(plugin_uri)
#k['param_map'].append(param_map)
#k['stereo'].append(stereo)
#k['set_nontrainable_parameters'].append(set_nontrainable_parameters)
#k['new_parameter_range'].append(new_parameter_range)
#



# # DGEQ10
#plugin_uri = 'https://gitlab.com/dmt-soma/lv2_plugs#SexyZebra10'
#stereo = False
#thresholds = list(range(0, 60, 6))
#ratios = [x + 1 for x in thresholds]
#knees = [x + 2 for x in thresholds]
#attacks = [x + 3 for x in thresholds]
#releases = [x + 4 for x in thresholds]
#makeups = [x + 5 for x in thresholds]
#mastergain = [60]
#
#parameters =  ratios + attacks + releases + makeups + mastergain + thresholds
#
#set_nontrainable_parameters = dict.fromkeys(knees, 0)
#
#new_parameter_range = dict.fromkeys(ratios, [1, 30])
#new_parameter_range.update(dict.fromkeys(thresholds, [-130, 0]))
#
#params = len(parameters)
#param_map = {}
#for i, port in enumerate(parameters):
#    param_map[i] = port
#k['params'].append(params)
#k['plugin_uri'].append(plugin_uri)
#k['param_map'].append(param_map)
#k['stereo'].append(stereo)
#k['set_nontrainable_parameters'].append(set_nontrainable_parameters)
#k['new_parameter_range'].append(new_parameter_range)

####SexyZebra 30
plugin_uri = 'https://gitlab.com/dmt-soma/lv2_plugs#SexyZebra30'
stereo = False
thresholds = list(range(0, 180, 6))
ratios = [x + 1 for x in thresholds]
knees = [x + 2 for x in thresholds]
attacks = [x + 3 for x in thresholds]
releases = [x + 4 for x in thresholds]
makeups = [x + 5 for x in thresholds]
mastergain = [180]

parameters =  ratios + attacks + releases + makeups + mastergain + thresholds

set_nontrainable_parameters = dict.fromkeys(knees, 0)

new_parameter_range = dict.fromkeys(ratios, [1, 30])
new_parameter_range.update(dict.fromkeys(thresholds, [-130, 0]))

params = len(parameters)
param_map = {}
for i, port in enumerate(parameters):
    param_map[i] = port
k['params'].append(params)
k['plugin_uri'].append(plugin_uri)
k['param_map'].append(param_map)
k['stereo'].append(stereo)
k['set_nontrainable_parameters'].append(set_nontrainable_parameters)
k['new_parameter_range'].append(new_parameter_range)

