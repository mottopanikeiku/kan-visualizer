"use strict";

// Invoked by the unittest suite; model JSON and fixed inputs arrive on stdin.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const evaluator = require("../web/js/model-forward.js");
const payload = JSON.parse(fs.readFileSync(0, "utf8"));
const ATOL = 1e-6;
const RTOL = 1e-6;
let curveMaxAbsoluteError = 0;
let curvesChecked = 0;

function close(actual, expected, label, atol = ATOL, rtol = RTOL) {
    assert.ok(Number.isFinite(actual), `${label}: non-finite actual value`);
    assert.ok(Number.isFinite(expected), `${label}: non-finite expected value`);
    const error = Math.abs(actual - expected);
    assert.ok(error <= atol + rtol * Math.abs(expected),
        `${label}: ${actual} != ${expected}, absolute error ${error}`);
    return error;
}

const results = payload.cases.map(({model, inputs, name}) => {
    for (const layer of model.layers) {
        assert.equal(layer.edge_evaluations.length,
            layer.input_features * layer.output_features, `${name}: every edge has a curve`);
        const indices = new Set();
        for (const curve of layer.edge_evaluations) {
            const key = `${curve.output_idx},${curve.input_idx}`;
            assert.ok(!indices.has(key), `${name}: duplicate edge curve ${key}`);
            indices.add(key);
            assert.equal(curve.x_values.length, curve.y_values.length);
            assert.ok(curve.x_values.length > 2, `${name}: sampled curve is nonempty`);
            curve.x_values.forEach((x, index) => {
                const value = evaluator.edgeValue(layer, curve.input_idx, curve.output_idx, x);
                curveMaxAbsoluteError = Math.max(curveMaxAbsoluteError,
                    close(value, curve.y_values[index], `${name}: exported edge curve ${key}`));
                curvesChecked += 1;
            });
        }
    }
    return inputs.map(input => evaluator.forward(model, input));
});

// Unequal coordinates matter: a one-coordinate fallback or swapped axes must fail.
for (const input of [[0.4, -0.7], [-1.3, 0.2], [2.25, -1.75], [0, 1.1]]) {
    close(evaluator.targetValue("2d_gaussian", input),
        Math.sin(input[0]) * Math.exp(-(input[1] ** 2)), "2D Gaussian target", 1e-12, 1e-12);
    close(evaluator.targetValue("2d_complex", input),
        Math.sin(input[0] * input[1]) + 0.5 * Math.tanh(input[0] - input[1]),
        "2D complex target", 1e-12, 1e-12);
}
for (const x of [-2.4, -0.13, 0, 0.71, 2.1]) {
    close(evaluator.targetValue("1d_sine_wave", [x]),
        Math.sin(3 * x) + 0.3 * Math.cos(10 * x), "1D target", 1e-12, 1e-12);
}

for (const dataset of Object.values(payload.datasets)) {
    for (const sample of dataset.samples) {
        close(evaluator.targetValue(dataset.target_id, sample.input), sample.output,
            `exported target sample ${dataset.target_id}`);
    }
}

const firstModel = payload.cases[0].model;
const validInput = new Array(firstModel.metadata.architecture[0]).fill(0.2);
const clone = () => JSON.parse(JSON.stringify(firstModel));
let invalid = clone();
invalid.metadata.basis = "cubic_spline";
assert.throws(() => evaluator.forward(invalid, validInput), "unsupported basis must fail");
invalid = clone();
invalid.layers[0].base_activation = "unsupported";
assert.throws(() => evaluator.forward(invalid, validInput), "unsupported activation must fail");
assert.throws(() => evaluator.edgeValue(invalid.layers[0], 0, 0, 0.2));
invalid = clone();
delete invalid.layers[0].rbf_coefficients;
assert.throws(() => evaluator.forward(invalid, validInput), "missing coefficients must fail");
assert.throws(() => evaluator.forward(firstModel, []), "wrong input dimension must fail");
assert.throws(() => evaluator.targetValue("unknown_target", [0.2, -0.4]));
assert.throws(() => evaluator.targetValue("2d_gaussian", [0.2]));

process.stdout.write(JSON.stringify({results, curveMaxAbsoluteError, curvesChecked}));
