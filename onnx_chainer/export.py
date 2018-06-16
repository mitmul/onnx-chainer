from __future__ import print_function

import collections
import heapq

import chainer
from chainer import function_node
from chainer import variable
import numpy
import onnx
from onnx.mapping import NP_TYPE_TO_TENSOR_TYPE

from onnx_chainer import functions

try:
    from onnx import checker
    from onnx import helper
    from onnx import numpy_helper

    _available = True
except ImportError:
    _available = False


def _check_available():
    if not _available:
        raise ImportError(
            'ONNX is not installed on your environment. Exporting your model '
            'in ONNX format needs the onnx package.\n\n'
            '\t$ pip install onnx\n\n')


def convert_parameter(parameter):
    if isinstance(parameter, chainer.Parameter):
        array = parameter.array
    elif isinstance(parameter, chainer.Variable):
        array = parameter.array
    elif isinstance(parameter, numpy.ndarray):
        array = parameter
    if array.shape == ():
        array = array[None]
    return numpy_helper.from_array(array, str(id(parameter)))


def create_node(
        func_name, cand, input_names, output_names, parameters):
    converter_name = 'convert_{}'.format(func_name)
    if hasattr(functions, converter_name):
        converter = getattr(functions, converter_name)
        nodes = converter(
            cand, input_names, output_names, parameters)
    else:
        raise ValueError('{} is not supported.'.format(func_name))
    for node in nodes:
        checker.check_node(node)
    return nodes


class ONNXExport(chainer.function_hook.FunctionHook):

    def __init__(self):
        self.graph = []
        self.additional_parameters = []
        self.network_inputs = {}

    def backward_postprocess(self, function, in_data, out_grad):
        if isinstance(function, chainer.function.FunctionAdapter):
            function = function.function
        func_name = function.__class__.__name__
        input_names = []
        for i in function.inputs:
            # Check if it is a Parameter object
            var = i.get_variable_or_none()
            if var is None:
                input_names.append(str(id(i)))  # Use VariableNode
                if str(id(i)) not in self.network_inputs:
                    self.network_inputs[str(id(i))] = i
            else:
                if isinstance(var, chainer.Parameter):
                    input_names.append(str(id(var)))
                else:
                    input_names.append(str(id(var)))
                    self.network_inputs[str(id(var))] = var
        output_names = [str(id(o())) for o in function.outputs]
        nodes = create_node(
            func_name, function, input_names, output_names,
            self.additional_parameters)
        for node in nodes:
            if node not in self.graph:
                self.graph.append(node)


def export(model, args, filename=None, export_params=True,
           graph_name='Graph', save_text=False):
    """Export function for chainer.Chain in ONNX format.

    This function performs a forward computation of the given
    :class:`~chainer.Chain`, ``model``, by passing the given argments ``args``
    directly. It means, the output :class:`~chainer.Variable` object ``y`` to
    make the computational graph will be created by:

    y = model(*args)

    Args:
        model (~chainer.Chain): The model object you want to export in ONNX
            format. It should have :meth:`__call__` method because the second
            argment ``args`` is directly given to the model by the ``[]``
            accessor.
        args (list or dict): The argments which are given to the model
            directly.
        filename (str or file-like object): The filename used for saving the
            resulting ONNX model. If None, nothing is saved to the disk.
        export_params (bool): If True, this function exports all the parameters
            included in the given model at the same time. If False, the
            exported ONNX model doesn't include any parameter values.
        graph_name (str): A string to be used for the ``name`` field of the
            graph in the exported ONNX model.
        save_text (bool): If True, the text format of the output ONNX model is
            also saved with ``.txt`` extention.

    Returns:
        A ONNX model object.

    """

    _check_available()

    chainer.config.enable_backprop = True

    model.to_cpu()

    # Convert all args into Variable objects
    args = list(args) if isinstance(args, (list, tuple)) else [args]
    for i, arg in enumerate(args):
        if not isinstance(arg, chainer.Variable):
            args[i] = chainer.Variable(arg)

    # Forward computation
    if isinstance(args, list):
        outputs = model(*args)
    elif isinstance(args, dict):
        outputs = model(**args)
    else:
        raise ValueError(
            'The \'args\' argument should be a list or dict. But a {} '
            'object was given.'.format(type(args)))

    initializers = []
    input_tensors = []
    for param in model.params():
        initializers.append(convert_parameter(param))
        param_shape = (1,) if param.shape == () else param.shape
        input_tensors.append(helper.make_tensor_value_info(
            str(id(param)), NP_TYPE_TO_TENSOR_TYPE[param.array.dtype], param_shape))

    with ONNXExport() as o:
        if isinstance(outputs, (list, tuple)):
            for output in outputs:
                output.grad = numpy.ones_like(
                    output.data, dtype=output.data.dtype)
                output.backward()
        elif isinstance(outputs, chainer.Variable):
            outputs.grad = numpy.ones_like(outputs.data)
            outputs.backward()

    # If additonal parameters are created during conversion
    if o.additional_parameters:
        for param in o.additional_parameters:
            initializers.append(convert_parameter(param))
            param_shape = (1,) if param.shape == () else param.shape
            input_tensors.append(helper.make_tensor_value_info(
                str(id(param)), NP_TYPE_TO_TENSOR_TYPE[param.array.dtype],
                param_shape))

    # Collect the network inputs
    for i in o.network_inputs.values():
        input_tensors.append(helper.make_tensor_value_info(
            str(id(i)), NP_TYPE_TO_TENSOR_TYPE[i.dtype], i.shape))

    # The graph must be topologically sorted
    graph = reversed(o.graph)

    # Convert output tensors
    output_tensors = []
    if isinstance(outputs, dict):
        outputs = list(outputs.values())
    if not isinstance(outputs, (list, tuple)):
        outputs = (outputs,)
    for output in outputs:
        output_tensors.append(helper.make_tensor_value_info(
            str(id(output)), NP_TYPE_TO_TENSOR_TYPE[output.dtype], output.shape))

    if not export_params:
        initializers = []

    onnx_graph = helper.make_graph(
        graph, graph_name, input_tensors, output_tensors,
        initializer=initializers)

    checker.check_graph(onnx_graph)

    model = helper.make_model(
        onnx_graph,
        producer_name='Chainer',
        producer_version=chainer.__version__)

    model.ir_version = onnx.IR_VERSION

    checker.check_model(model)

    if filename is not None and isinstance(filename, str):
        with open(filename, 'wb') as fp:
            fp.write(model.SerializeToString())
        if save_text:
            with open(filename + '.txt', 'w') as fp:
                print(model, file=fp)
    elif hasattr(filename, 'write'):
        filename.write(model.SerializeToString())

    return model
