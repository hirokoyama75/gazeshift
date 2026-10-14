"""Restore original TF1 meta graphs, freeze inference outputs, export both eyes.

Copyright 2019 Chih-Fan Hsu. Derived work, BSD-3-Clause; see LICENSE.
This program runs only inside the Python 3.11 exporter environment.
"""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
import hashlib
import re
import platform
import pickle
import numpy as np
import tensorflow as tf
import tf2onnx
import onnx
from onnx import compose
from onnx import helper, numpy_helper
from scripts.assets import ROOT, acquire, sha256
from scripts.parity_fixture import materialize

tf.compat.v1.disable_eager_execution()
FLOW = "tower_0/warping_model/Tanh:0"
LIGHT = "tower_0/warping_model/lcm_module/Reshape_1:0"
SAMPLER = "tower_0/warping_model/apply_transformation/Reshape_1:0"
# Feed the original tower's slice outputs directly: the training meta graph
# splits a batch across two towers, while each inference eye has N=1.
INPUTS = ["tower_0/strided_slice:0", "tower_0/strided_slice_1:0", "tower_0/strided_slice_2:0"]


def restore(side):
    prefix = ROOT / f"weights/tf/flx/12/{side}/{side}"
    graph = tf.Graph()
    with graph.as_default():
        meta = tf.compat.v1.MetaGraphDef()
        meta.ParseFromString(prefix.with_suffix(".meta").read_bytes())
        # Constant inference mode keeps training batch-stat branches out of ONNX.
        for node in meta.graph_def.node:
            if node.name == "model_control/phase_train":
                node.op = "Const"
                node.attr.clear()
                node.attr["dtype"].type = tf.bool.as_datatype_enum
                node.attr["value"].tensor.CopyFrom(tf.make_tensor_proto(False))
        saver = tf.compat.v1.train.import_meta_graph(meta, clear_devices=True)
        config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1, device_count={"GPU": 0})
        session = tf.compat.v1.Session(graph=graph, config=config)
        saver.restore(session, str(prefix))
        missing = session.run(tf.compat.v1.report_uninitialized_variables())
        if len(missing):
            raise ValueError(f"Unrestored checkpoint variables: {missing}")
        variables = tf.compat.v1.global_variables()
        # No initialization is run: compare every restored variable with its source.
        reader = tf.train.load_checkpoint(str(prefix))
        for variable in variables:
            np.testing.assert_array_equal(session.run(variable), reader.get_tensor(variable.op.name))
    return session, graph, len(variables)


def rename_io(model, mapping):
    for node in model.graph.node:
        for values in (node.input, node.output):
            for i, name in enumerate(values):
                if name in mapping:
                    values[i] = mapping[name]
    for values in (model.graph.input, model.graph.output, model.graph.value_info):
        for value in values:
            value.name = mapping.get(value.name, value.name)
    return model


def align_conv_channels(model):
    """Zero-pad irregular input channels so ORT can select blocked CPU kernels.

    This is an exact layout optimization: every new weight is zero; original
    learned weights, graph connectivity, activations and outputs are preserved.
    """
    constants = {i.name: i for i in model.graph.initializer}
    nodes = []
    for node in model.graph.node:
        if node.op_type == "Conv" and node.input[1] in constants:
            weights = numpy_helper.to_array(constants[node.input[1]])
            channels = weights.shape[1]
            if channels > 4 and channels % 8:
                extra = 8 - channels % 8
                name = node.name + "_aligned"
                pads = np.array([0,0,0,0,0,extra,0,0], np.int64)
                model.graph.initializer.append(numpy_helper.from_array(pads, name+"_pads"))
                nodes.append(helper.make_node("Pad", [node.input[0], name+"_pads"], [name], name=name))
                node.input[0] = name
                padded = np.pad(weights, ((0,0),(0,extra),(0,0),(0,0)))
                constants[node.input[1]].CopyFrom(numpy_helper.from_array(padded, node.input[1]))
        nodes.append(node)
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    # Shape annotations for replaced intermediates are re-inferred.
    del model.graph.value_info[:]
    return onnx.shape_inference.infer_shapes(model)


def preserve_blocked_affines(model):
    """Express channel affine steps as equivalent depthwise 1x1 convolutions.

    ORT can keep these in its blocked CPU layout instead of inserting a reorder
    around each Mul/Add pair. No checkpoint parameter or activation is changed.
    """
    constants = {i.name: numpy_helper.to_array(i) for i in model.graph.initializer}
    producers = {output: node for node in model.graph.node for output in node.output}
    consumers = {}
    for node in model.graph.node:
        for name in node.input:
            consumers.setdefault(name, []).append(node)
    replacements, removed = {}, set()
    for node in model.graph.node:
        if node.op_type != "Add":
            continue
        multiply = next((producers[i] for i in node.input if i in producers and producers[i].op_type == "Mul"), None)
        bias = next((constants[i] for i in node.input if i in constants), None)
        if multiply is None or bias is None or len(consumers[multiply.output[0]]) != 1:
            continue
        scale = next((constants[i] for i in multiply.input if i in constants), None)
        source = next((i for i in multiply.input if i not in constants), None)
        if scale is None or source is None or scale.shape != bias.shape or scale.ndim != 4 or scale.shape[2:] != (1, 1):
            continue
        names = []
        for tag, value in (("scale", scale.reshape(-1, 1, 1, 1)), ("bias", bias.ravel())):
            name = node.name + "_" + tag
            model.graph.initializer.append(numpy_helper.from_array(value, name))
            names.append(name)
        replacements[node.name] = helper.make_node("Conv", [source, *names], list(node.output),
            name=node.name + "_affine", group=scale.size, kernel_shape=[1, 1])
        removed.add(multiply.name)
    nodes = [replacements.get(n.name, n) for n in model.graph.node if n.name not in removed]
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    used = {i for n in nodes for i in n.input}
    keep = [i for i in model.graph.initializer if i.name in used]
    del model.graph.initializer[:]
    model.graph.initializer.extend(keep)
    del model.graph.value_info[:]
    return onnx.shape_inference.infer_shapes(model)


def static_lighting_softmax(model):
    """ONNX supports the original channel softmax directly on the 4D tensor.

    TF1's automatic flatten/restore workaround adds redundant dynamic Shape
    paths; removing it also removes their nondeterministic converter aliases.
    """
    nodes = list(model.graph.node)
    producers = {output: node for node in nodes for output in node.output}
    for side in ("l", "r"):
        output = f"lcm_{side}"
        after = producers[output]
        softmax = producers[after.input[0]]
        before = producers[softmax.input[0]]
        if (after.op_type, softmax.op_type, before.op_type) != ("Reshape", "Softmax", "Reshape"):
            raise ValueError("Unexpected original lighting softmax structure")
        nodes.append(helper.make_node("Softmax", [before.input[0]], [output], name=f"{side}/lighting_softmax", axis=3))
    producers = {output: node for node in nodes for output in node.output}
    ordered, seen = [], set()
    def visit(value):
        node = producers.get(value)
        if node is None or node.name in seen:
            return
        seen.add(node.name)
        for source in node.input:
            visit(source)
        ordered.append(node)
    for output in model.graph.output:
        visit(output.name)
    del model.graph.node[:]
    model.graph.node.extend(ordered)
    used = {source for node in ordered for source in node.input}
    keep = [value for value in model.graph.initializer if value.name in used]
    del model.graph.initializer[:]
    model.graph.initializer.extend(keep)
    del model.graph.value_info[:]
    return onnx.shape_inference.infer_shapes(model)


def canonical_constants(model):
    """Content-based names and ordering make fresh exports byte reproducible."""
    aliases, values = {}, {}
    for initializer in model.graph.initializer:
        array = np.ascontiguousarray(numpy_helper.to_array(initializer))
        digest = hashlib.sha256(array.dtype.str.encode() + repr(array.shape).encode() + array.tobytes()).hexdigest()
        name = "constant_" + digest
        aliases[initializer.name] = name
        values[name] = numpy_helper.from_array(array, name)
    for node in model.graph.node:
        for index, name in enumerate(node.input):
            node.input[index] = aliases.get(name, name)
    del model.graph.initializer[:]
    model.graph.initializer.extend(values[name] for name in sorted(values))
    return model


def stable_converter_symbols(model):
    # Converter counters vary when equivalent rewrites are visited in a
    # different order. Retain the readable source path, replacing that suffix.
    def stable(name):
        return re.sub(r"__\d+(?=:|$)", "__converted", name)
    for node in model.graph.node:
        node.name = stable(node.name)
        for values in (node.input, node.output):
            for index, name in enumerate(values):
                values[index] = stable(name)
    for values in (model.graph.input, model.graph.output, model.graph.value_info):
        for value in values:
            value.name = stable(value.name)
    names = [node.name for node in model.graph.node]
    outputs = [value for node in model.graph.node for value in node.output]
    if len(names) != len(set(names)) or len(outputs) != len(set(outputs)):
        raise ValueError("Converter symbol normalization would alias distinct operations")
    return model


def main():
    if sys.version_info[:2] != (3, 11) or tf.__version__ != "2.15.0" or tf2onnx.__version__ != "1.17.0":
        raise RuntimeError("Exporter requires Python 3.11 / TF 2.15.0 / tf2onnx 1.17.0")
    hashes = acquire()
    fixture = materialize()
    references = []
    models, restoration = [], {}
    for side in ("L", "R"):
        s = side.lower()
        session, graph, count = restore(side)
        restoration[side] = count
        outputs = [graph.get_tensor_by_name(name) for name in (FLOW, LIGHT)]
        # The sampler output is found from the original graph's final reshape.
        sampler_ops = [op for op in graph.get_operations() if op.name.startswith("tower_0/warping_model/apply_transformation/") and op.type == "Reshape"]
        sampler = sampler_ops[-1].outputs[0]
        for index, case in enumerate(fixture["cases"]):
            if side == "L":
                references.append({"label": case["label"], "outputs": {}})
            feed = {graph.get_tensor_by_name(n): case["inputs"][f"{k}_{s}"] for n, k in zip(INPUTS, ("img", "fp", "ang"))}
            flow, lighting, warped = session.run([*outputs, sampler], feed)
            references[index]["outputs"].update({f"flow_{s}": flow, f"lcm_{s}": lighting, f"sampler_{s}": warped})
        with graph.as_default():
            frozen = tf.compat.v1.graph_util.convert_variables_to_constants(session, graph.as_graph_def(), [FLOW[:-2], LIGHT[:-2]])
        shapes = dict(zip(INPUTS, ([1, 48, 64, 3], [1, 48, 64, 12], [1, 2])))
        model, _ = tf2onnx.convert.from_graph_def(frozen, input_names=INPUTS, output_names=[FLOW, LIGHT], shape_override=shapes, opset=15)
        model = compose.add_prefix(model, f"{s}/")
        mapping = {f"{s}/{name}": f"{kind}_{s}" for name, kind in zip(INPUTS, ("img", "fp", "ang"))}
        mapping.update({f"{s}/{FLOW}": f"flow_{s}", f"{s}/{LIGHT}": f"lcm_{s}"})
        models.append(rename_io(model, mapping))
        session.close()
    combined = compose.merge_models(models[0], models[1], io_map=[])
    combined = onnx.shape_inference.infer_shapes(combined)
    for output in combined.graph.output:
        for dimension, size in zip(output.type.tensor_type.shape.dim, (1, 48, 64, 2)):
            dimension.dim_value = size
    combined = align_conv_channels(combined)
    combined = preserve_blocked_affines(combined)
    combined = static_lighting_softmax(combined)
    combined = canonical_constants(combined)
    combined = stable_converter_symbols(combined)
    onnx.checker.check_model(combined)
    if any("apply_transformation" in node.name or node.op_type in ("Floor", "GatherND") for node in combined.graph.node):
        raise ValueError("Sampler leaked into exported ONNX graph")
    model_path = ROOT / "weights/flx_12.onnx"
    onnx.save(combined, model_path)
    ref_path = ROOT / ".cache/parity/restored_tf.pkl"
    ref_path.write_bytes(pickle.dumps({"schema": 1, "cases": references, "model_sha256": sha256(model_path), "fixture_sha256": sha256(ROOT / '.cache/parity/dirl_flx12.pkl')}, protocol=4))
    report = {"status": "PASS", "python": platform.python_version(), "tensorflow": tf.__version__, "tf2onnx": tf2onnx.__version__,
              "opset": 15, "restored_variables": restoration, "hashes": hashes,
              "model_sha256": sha256(model_path), "reference_sha256": sha256(ref_path),
              "fixture_sha256": sha256(ROOT / ".cache/parity/dirl_flx12.pkl"), "cases": len(references)}
    report["transformations"] = ["zero_channel_padding", "depthwise_affines", "static_lighting_softmax", "deterministic_constants", "stable_converter_symbols"]
    (ROOT / "artifacts/export.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
