import tensorflow as tf

vers = (tf.__version__).split(".")
if int(vers[0]) == 2 or int(vers[0]) == 1 and int(vers[1]) > 12:
    tf = tf.compat.v1
else:
    tf = tf


def read_graph(file):
    with tf.io.gfile.GFile(file, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        return graph_def


def finalize_graph(graph_def):
    graph = tf.Graph()
    with graph.as_default():
        tf.import_graph_def(graph_def, name="DLC")
    graph.finalize()

    return graph


def get_output_nodes(graph):
    op_names = [str(op.name) for op in graph.get_operations()]
    if "concat_1" in op_names[-1]:
        output = [op_names[-1]]
    else:
        output = [op_names[-1], op_names[-2]]

    return output


def get_output_tensors(graph):
    output_nodes = get_output_nodes(graph)
    output_tensor = [out + ":0" for out in output_nodes]
    return output_tensor


def get_input_tensor(graph):
    input_tensor = str(graph.get_operations()[0].name) + ":0"
    return input_tensor


def extract_graph(graph, tf_config=None):
    input_tensor = get_input_tensor(graph)
    output_tensor = get_output_tensors(graph)
    sess = tf.Session(graph=graph, config=tf_config)
    inputs = graph.get_tensor_by_name(input_tensor)
    outputs = [graph.get_tensor_by_name(out) for out in output_tensor]

    return sess, inputs, outputs
