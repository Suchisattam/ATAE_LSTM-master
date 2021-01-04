import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def colorize(words, color_array , aspect):
    # words is a list of words
    # color_array is an array of numbers between 0 and 1 of length equal to words
    cmap = matplotlib.cm.get_cmap('YlOrRd')
    template = '<span class="barcode"; style="color: black; background-color: {}">{}</span>'
    colored_string = ''
    for word, color in zip(words, color_array):
        color = matplotlib.colors.rgb2hex(cmap(color)[:3])
        colored_string += template.format(color, '&nbsp' + word + '&nbsp')
    colored_string += '_______________' + aspect
    colored_string += '<br />'
    colored_string += '<br />'
    return colored_string


## RESTAURANT SENTENCES AND ATTENTION WEIGHTS
rest_data_length = 5   
rest_raw_texts = [
    'The menu had a lot of varieties', 
    'the fajita we tried was tasteless and burnt and the sauce was way too sweet.',
    'The service is terrible but the food was good' ,
    'The service is terrible but the food was good', 
    'they have one of the fastest delivery times in the city',]

rest_aspects = ['cheese' , 'taste' , 'food' , 'service' , 'service' ]

rest_attention_weights = [
    [4.0942e-04, 2.1552e-02, 1.6689e-02, 9.3163e-03, 2.6084e-02,1.4917e-01, 7.7652e-01, 3.2482e-05, 3.2482e-05, 3.2482e-05,3.2482e-05, 3.2482e-05, 3.2482e-05, 3.2482e-05, 3.2482e-05],
    [3.8567e-04, 6.9086e-04, 8.2347e-04, 5.7094e-04, 4.4926e-04,4.8126e-01, 3.3169e-02, 8.5434e-02, 6.1765e-02, 3.0310e-02,6.4746e-02, 3.0864e-02, 1.2570e-01, 2.6315e-02, 5.7514e-02],
    [1.7922e-05, 6.5493e-04, 5.2533e-03, 1.7930e-06, 1.7930e-06,4.4833e-02, 2.3628e-01, 7.4692e-02, 2.2821e-01, 1.7930e-06,1.7930e-06, 1.7930e-06, 1.9987e-01, 2.1018e-01, 1.7930e-06],
    [2.3875e-05, 5.5896e-04, 4.1451e-03, 3.2001e-01, 1.6795e-01,3.0673e-02, 2.3028e-01, 8.2269e-02, 1.6405e-01, 6.6525e-06,6.6525e-06, 6.6525e-06, 6.6525e-06, 6.6525e-06, 6.6525e-06],
    [1.3852e-04, 5.9937e-05, 9.5658e-05, 5.0536e-04, 7.3846e-04,6.5637e-03, 2.1863e-02, 7.9633e-02, 3.0781e-01, 2.4224e-01,3.4028e-01, 1.7187e-05, 1.7187e-05, 1.7187e-05, 1.7187e-05]]

## LAPTOP SENTENCES AND ATTENTION WEIGHTS
laptop_data_length = 5   
laptop_raw_texts = [
    'Laptop has High Definition display which give amazing real life experience while watching movies, but the webcam picture quality is very bad',
    'Though my system is working fine even after 5 years of purchasing it, sometimes software gets unresponsive', 
    'Though my system is working fine even after 5 years of purchasing it, sometimes software gets unresponsive', 
    'The laptop is attractive for youngsters because of the features it offers' , 
    'Laptop loads the operating system very fast maybe due to presence of 256 Solid State drive']

laptop_aspects = ['display' , 'software' , 'system' , 'features' ,'speed']

laptop_attention_weights = [
    [3.7177e-06, 1.5710e-05, 3.5640e-05, 1.1776e-05, 3.1754e-01,6.7124e-06, 1.0679e-04, 3.0267e-01, 1.5422e-01, 1.4967e-01,5.1496e-02, 3.6772e-03, 1.6586e-02, 4.7939e-04, 8.6450e-05,1.3660e-06, 1.5650e-06, 1.6116e-05, 1.7245e-04, 1.1811e-04,2.9724e-03, 1.1149e-04],
    [7.0389e-05, 3.5769e-05, 1.0784e-05, 7.6491e-05, 1.8371e-03,8.0504e-02, 6.2054e-07, 1.1596e-02, 1.0786e-03, 3.4368e-03,2.3257e-03, 4.3862e-03, 9.0529e-03, 6.7243e-02, 1.3362e-03,2.9015e-02, 6.7334e-01, 6.2054e-07, 6.2054e-07, 6.2054e-07,6.2054e-07, 1.1466e-01 ],
    [3.0634e-03, 9.6890e-04, 1.4699e-04, 6.2412e-04, 7.1332e-03,7.3058e-01, 1.9355e-01, 5.1322e-03, 2.5587e-04, 4.1122e-04,1.7151e-04, 4.5381e-04, 2.1177e-03, 1.0996e-02, 1.5680e-04,4.4639e-03, 3.9653e-02, 2.4867e-05, 2.4867e-05, 2.4867e-05,2.4867e-05, 2.4867e-05],
    [7.9431e-06, 3.2482e-05, 1.7622e-04, 5.1869e-01, 1.2307e-01,7.2585e-02, 1.6731e-01, 7.2312e-03, 5.5850e-04, 1.9675e-02,4.7059e-03, 8.5837e-02, 1.1076e-05, 1.1076e-05, 1.1076e-05,1.1076e-05, 1.1076e-05, 1.1076e-05, 1.1076e-05, 1.1076e-05,1.1076e-05, 1.1076e-05],
    [9.5147e-07, 4.6032e-05, 1.0215e-05, 1.1854e-06, 3.0445e-06,6.5701e-04, 8.2794e-01, 9.8149e-02, 7.1890e-02, 6.7341e-05,3.3819e-05, 6.5263e-06, 1.9073e-05, 7.1407e-04, 2.3318e-04,2.3043e-04, 2.2947e-07, 2.2947e-07, 2.2947e-07, 2.2947e-07,2.2947e-07, 2.2947e-07]]

# words = 'they have one of the fastest delivery times in the city'.split()
# color_array = [1.3852e-04, 5.9937e-05, 9.5658e-05, 5.0536e-04, 7.3846e-04,6.5637e-03, 2.1863e-02, 7.9633e-02, 3.0781e-01, 2.4224e-01, 3.4028e-01, 1.7187e-05, 1.7187e-05, 1.7187e-05, 1.7187e-05][0:len(words)]
for i in range(0,rest_data_length): 
    words = rest_raw_texts[i]
    color_array = rest_attention_weights[i]
    aspect = rest_aspects[i]

    words = words.split()
    color_array = color_array[0:len(words)]
    
    s = colorize(words, color_array,aspect)
    with open('colorize.html', 'a') as f:
        f.write(s)

for i in range(0,laptop_data_length): 
    words = laptop_raw_texts[i]
    color_array = laptop_attention_weights[i]
    aspect = laptop_aspects[i]

    words = words.split()
    color_array = color_array[0:len(words)]
    
    s = colorize(words, color_array,aspect)
    with open('colorize.html', 'a') as f:
        f.write(s)




# to display in ipython notebook
from IPython.display import display, HTML
display(HTML(s))

# or simply save in an html file and open in browser

f.close()