'''
@copyright: Copyright (c)  2016-2017, Vignesh Srinivasan, Sebastian Lapuschkin, Alexander Binder, Gregoire Montavon, Klaus-Robert Mueller, Wojciech Samek
@license : BSD-2-Clause
'''

import render
import numpy as np
import relevance_methods

class relevance_analyzer(object):

    def __init__(self, results, weights, biases, epoch_id, config):

        relevance_method = config.get('visualization', 'visualization_method')
        epsilon_value = float(config.get('visualization', 'epsilon_value'))
        alpha_value = float(config.get('visualization', 'alpha_value'))
        self.pixel_scaling_factor = int(config.get('visualization', 'pixel_scaling_factor'))
        extension = config.get('visualization', 'extension')
        self.image_dir = config.get('directories', 'exp_dir') + '/heat_map_image_dir'

        for i in range(1, 4):
             
            #inputs = results[i]
            #weight = weights[i]
            #bias = biases[i]
            #outputs = results[i+1]

            if relevance_method == "simple_lrp":
                relevance = relevance_methods.simple_lrp(results[i+1], results[i], biases[i], weights[i])
            elif relevance_method == "flat_lrp":
                relevance = relevance_methods.flat_lrp(results[i+1], results[i], biases[i], weights[i])
            elif relevance_method == "ww_lrp":
                relevance = relevance_methods.ww_lrp(results[i+1], results[i], biases[i], weights[i])
            elif relevance_method == "epsilon_lrp":
                relevance = relevance_methods.simple_lrp(results[i+1], results[i], biases[i], weights[i], epsilon_value)
            elif relevance_method == "alphabeta_lrp":
                relevance = relevance_methods.alphabeta_lrp(results[i+1], results[i], biases[i], weights[i], alpha_value)
            
            image_name_last = '_epoch_'+str(epoch_id)+'_layer_'+str(i)+extension

            print 'Completed relevance calculation for ' + image_name_last
            
            self.create_image(results[i], relevance, image_name_last)
    

    def create_image(self, input_value, relevance_value, image_name_last):
    
        input_image = input_value.reshape([1,input_value.shape[0],input_value.shape[1],1])
        relevance_image = relevance_value.reshape([1,relevance_value.shape[0],relevance_value.shape[1],1])

        self.visualize(relevance_image, input_image, relevance_value.shape, self.image_dir+'/relevance_on_input'+ image_name_last)

        self.visualize(relevance_image, None, relevance_value.shape, self.image_dir+'/relevance_only'+ image_name_last)

        #self.visualize(input_image, None, input_value.shape, self.image_dir+'/input_only'+ image_name_last)

        #self.visualize(input_image, relevance_image, input_value.shape, self.image_dir+'/input_on_relevance'+ image_name_last)


    def visualize(self, relevances, images_tensor, shape, image_name):

        #n,w,h, dim = relevances.shape
        #heatmaps = []
        if images_tensor is not None:
            assert relevances.shape==images_tensor.shape, 'Relevances shape != Images shape'
        for h,heat in enumerate(relevances):
            if images_tensor is not None:
                input_image = images_tensor[h]
                maps = render.hm_to_rgb(heat, input_image, scaling = self.pixel_scaling_factor, sigma = 2, shape = shape, name=image_name)
            else:
                maps = render.hm_to_rgb(heat, scaling = self.pixel_scaling_factor, sigma = 2, shape = shape, name=image_name)
            #heatmaps.append(maps)
        #R = np.array(heatmaps)
        print 'shape of image: '
        print maps.shape

