from collections import OrderedDict
from utee import misc, quant
import numpy as np

def quant_model_params(model, quant_method='linear', param_bits=8, bn_bits=32, overflow_rate=0.0):
    ''' 
    quantize parameters
    '''
    if param_bits < 32:
        state_dict = model.state_dict()
        state_dict_quant = OrderedDict()
        for i, (k, v) in enumerate(state_dict.items()):
            if 'running' in k:
                if bn_bits >=32:
                    # print("Ignoring {}".format(k))
                    # print(v.data.cpu().numpy().shape)
                    state_dict_quant[k] = v
                    continue
                else:
                    bits = bn_bits
            elif 'num_batches_tracked' in k:
                # print("Ignoring {}".format(k))
                # print(v.data.cpu().numpy().shape)
                state_dict_quant[k] = v
                continue
            else:
                bits = param_bits

            # print(i, k, bits)
            # print('v:', type(v.data.cpu().numpy()), v.data.cpu().numpy().shape)

            if quant_method == 'linear':
                sf = bits - 1. - quant.compute_integral_part(v, overflow_rate=overflow_rate)
                v_quant  = quant.linear_quantize(v, sf, bits=bits)
            elif quant_method == 'log':
                v_quant = quant.log_minmax_quantize(v, bits=bits)
            elif quant_method == 'minmax':
                v_quant = quant.min_max_quantize(v, bits=bits)
            else:
                v_quant = quant.tanh_quantize(v, bits=bits)
            state_dict_quant[k] = v_quant

            if k in ['linear.weightB', 'layer4.2.conv1.weightB']:
                np.savetxt('./%s.txt' % k, v.data.cpu().numpy().squeeze())
                np.savetxt('./%s_quant.txt' % k, v_quant.data.cpu().numpy().squeeze())
            
        model.load_state_dict(state_dict_quant)

    else:
        pass

    return model