"""
    A sequence normalization flow framework with memory saving, automatic Jacobian tracking, and object-oriented programming features.
        1. Memory saving. It has memory saving properties within and between blocks, just as simple as the memcnn library.
        2. Automatic Jacobian tracking. Based on custom modules, automatic computation of log Jacobian determinant is implemented, just as simple as the FrEIA library.
        3. Object-oriented programming. Using Python's object-oriented programming features, it's easy to construct reversible neural networks based on custom components.

    To best of our knowledge, this is the first normalization flow (reversible neural network) framework that implements memory saving and automatic Jacobian tracking.
    The entire framework consists of only one file, autoFlow.py, which is easy to use and requires no installation!

    Note: the autoFlow framework is the core framework used in our work (PyramidFlow, CVPR 2023), which is more powerful and user-friendly than memcnn or FrEIA. If it is helpful, please cite our work.
    @article{lei2023pyramidflow,
        title={PyramidFlow: High-Resolution Defect Contrastive Localization using Pyramid Normalizing Flow},
        author={Jiarui Lei and Xiaobo Hu and Yue Wang and Dong Liu},
        journal={CVPR},
        year={2023}
    }
"""


import torch
import torch.nn as nn
from typing import Tuple
from abc import abstractmethod
from torch.cuda.amp import custom_fwd, custom_bwd

__all__ = ["InvertibleModule", "SequentialNF", "SequentialNet"]


class InvertibleModule(nn.Module):
    """
        Base class for constructing normalizing flow.
        You should be implemente `forward` and `inverse` function mannuly, which define a basic invertible module.
        Each function needs to implement the corresponding output tensor (the value returned by the function) for a given input tensor, 
            and also needs to implement the Jacobian determinant of the output tensor relative to the input tensor.

        Note: function `_forward` and `_inverse` is hidden function for user, should not be modified or called.
    """
    def __init__(self):
        super(InvertibleModule, self).__init__()
    """ Abstract functions (`forward` and `inverse`) need to be implemented explicitly. """
    @abstractmethod
    def forward(self, inputs: Tuple[torch.Tensor], logdets: Tuple[torch.Tensor]):
        raise NotImplementedError
    @abstractmethod
    def inverse(self, outputs: Tuple[torch.Tensor], logdets: Tuple[torch.Tensor]):
        raise NotImplementedError

    """ Hidden functions (`_forward` and `_inverse`) for implementing SequentialNF. """
    def _forward(self, *inputs_logdets):
        assert len(inputs_logdets)%2 == 0
        inputs  = inputs_logdets[:len(inputs_logdets)//2]
        logdets = inputs_logdets[len(inputs_logdets)//2:]
        outputs, logdets = self.forward(inputs, logdets)
        return outputs + logdets # Kept in a repeatable form.
    def _inverse(self, *outputs_logdets):
        assert len(outputs_logdets)%2 == 0
        outputs = outputs_logdets[:len(outputs_logdets)//2]
        logdets = outputs_logdets[len(outputs_logdets)//2:]
        inputs, logdets = self.inverse(outputs, logdets)
        return inputs + logdets



class AutoNFSequential(torch.autograd.Function):
    """ 
        Automatic implementation for sequential normalizing flows. 
        This class is hidden class for user, should not be modified or called.
    """
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, _forward_lst, _inverse_lst, inplogsRange, paramsRanges, *inplogs_and_params):# parameter passing only by *inplogs_and_params
        assert inplogsRange[1]%2 == 0
        inplogs = inplogs_and_params[inplogsRange[0]: inplogsRange[1]] # Save for the gradient later

        with torch.no_grad():
            outlogs = tuple([inplog.detach() for inplog in inplogs])
            for _forward in _forward_lst:
                outlogs = _forward(*outlogs) # Not save forward tensor
                for outlog in outlogs:
                    assert not outlog.isnan().any()

        ctx._forward_lst = _forward_lst
        ctx._inverse_lst = _inverse_lst
        ctx.outlogsRange = inplogsRange
        ctx.paramsRanges = paramsRanges

        outlogs = tuple([outlog.detach() for outlog in outlogs])
        params = inplogs_and_params[inplogsRange[1]:]
        ctx.save_for_backward(*outlogs, *params) # only the last output

        return outlogs

    @staticmethod
    @custom_bwd
    def backward(ctx, *grad_outlogs): 
        if not torch.autograd._is_checkpoint_valid():
            raise RuntimeError("This function is not compatible with .grad(), please use .backward() if possible")
        outlogs_params = ctx.saved_tensors
        outlogs = outlogs_params[ctx.outlogsRange[0]: ctx.outlogsRange[1]]
        params = [outlogs_params[range[0]: range[1]] for range in ctx.paramsRanges]
        _inverse_lst = ctx._inverse_lst
        _forward_lst = ctx._forward_lst
        grad_outlogs_loop = grad_outlogs

        # While reverse calculation, calculate gradient.There is always only one hidden tensor.
        grad_params = tuple() # Saved parameter gradients.
        detached_outlogs_loop = tuple([outlog.detach() for outlog in outlogs])
        for _forward, _inverse, param in zip(reversed(_forward_lst), reversed(_inverse_lst), reversed(params)):
            with torch.no_grad():
                inplogs = _inverse(*detached_outlogs_loop)
            with torch.set_grad_enabled(True):
                inplogs_loop = tuple([inplog.detach().requires_grad_() for inplog in inplogs])
                outlogs_loop = _forward(*inplogs_loop)
            grad_inplogs_params = torch.autograd.grad(outputs=outlogs_loop, grad_outputs=grad_outlogs_loop, inputs=inplogs_loop+param )

            detached_outlogs_loop = tuple([inplog.detach() for inplog in inplogs]) 
            grad_outlogs_loop = grad_inplogs_params[ :len(inplogs_loop)] # The gradient of the input
            grad_params = grad_inplogs_params[len(inplogs_loop): ] + grad_params # The gradient of the parameter
        
        grad_inplogs = grad_outlogs_loop
        return (None, None, None, None, ) + grad_inplogs + grad_params


class SequentialNF(InvertibleModule):
    """ A constructor class to build memory saving normalizing flows by a tuple of `InvertibleModule` """
    def __init__(self, modules: Tuple[InvertibleModule]):
        super(SequentialNF, self).__init__()
        self.moduleslst = nn.ModuleList(modules)
        self._forward_lst = tuple([module._forward for module in modules])
        self._inverse_lst = tuple([module._inverse for module in modules])
        self.params = [[p for p in module.parameters() if p.requires_grad] for module in self.moduleslst] # to fix 

    def forward(self, inputs, logdets):# Calculate the Jacobian determinant of the output return value relative to the input value, 
                                        #   which is partial{output}/partial{input}
        assert len(inputs) == len(logdets)
        inplogsRange = [0, len(inputs)+len(logdets)]
        paramsRange, lastIdx = [], inplogsRange[-1]
        for param in self.params:
            paramsRange.append([lastIdx, lastIdx+len(param)])
            lastIdx += len(param)
        
        outlogs = AutoNFSequential.apply(
            self._forward_lst, self._inverse_lst,
            inplogsRange, paramsRange,
            *inputs, *logdets,
            *[p for param in self.params for p in param]
        )
        mid = len(outlogs)//2
        return outlogs[ :mid], outlogs[mid: ]

    def inverse(self, outputs, logdets):# which is partial{input}/partial{output}
        assert len(outputs) == len(logdets)
        outlogsRange = [0, len(outputs)+len(logdets)]
        paramsRange, lastIdx = [], outlogsRange[-1]
        for param in self.params:
            paramsRange.append([lastIdx, lastIdx+len(param)])
            lastIdx += len(param)

        inplogs = AutoNFSequential.apply(
            list(reversed(self._inverse_lst)), list(reversed(self._forward_lst)),
            outlogsRange, paramsRange,
            *outputs, *logdets,
            *[p for param in self.params for p in param]
        )
        mid = len(inplogs)//2
        return inplogs[ :mid], inplogs[mid: ]


class SequentialNet(nn.Module):
    """ A constructor class to build pytorch-based normalizing flows by a tuple of `nn.Module` """ 
    def __init__(self, modules: Tuple[nn.Module]):
        super(SequentialNet, self).__init__()
        self.moduleslst = nn.ModuleList(modules)

    def forward(self, inputs, logdets):
        outputs = tuple(inputs)
        for  m in self.moduleslst:
            outputs, logdets = m(outputs, logdets)
        return outputs, logdets 

