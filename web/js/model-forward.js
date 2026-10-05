// Pure evaluator for the exported Gaussian-RBF KAN schema.
(function (root, factory) {
    const api = factory();
    if (typeof module === 'object' && module.exports) module.exports = api;
    else root.KANForward = api;
})(typeof globalThis !== 'undefined' ? globalThis : this, function () {
    'use strict';

    const activations = new Set(['silu', 'relu', 'gelu', 'tanh']);

    function requireValue(condition, message) {
        if (!condition) throw new Error(message);
    }

    function finite(value, name) {
        requireValue(Number.isFinite(value), `${name} must be a finite number`);
    }

    function matrix(value, rows, columns, name) {
        requireValue(Array.isArray(value) && value.length === rows, `${name}: wrong row count`);
        value.forEach(row => {
            requireValue(Array.isArray(row) && row.length === columns, `${name}: wrong column count`);
            row.forEach(x => finite(x, name));
        });
    }

    function validateLayer(layer) {
        requireValue(layer && Number.isInteger(layer.input_features) && layer.input_features > 0 &&
            Number.isInteger(layer.output_features) && layer.output_features > 0, 'Invalid layer dimensions');
        requireValue(activations.has(layer.base_activation), `Unsupported activation: ${layer.base_activation}`);
        requireValue(Array.isArray(layer.grid_points) && layer.grid_points.length >= 2, 'Missing Gaussian centers');
        layer.grid_points.forEach(x => finite(x, 'grid_points'));
        requireValue(Array.isArray(layer.grid_range) && layer.grid_range.length === 2 &&
            Number.isFinite(layer.grid_range[0]) && Number.isFinite(layer.grid_range[1]) &&
            layer.grid_range[0] < layer.grid_range[1], 'Invalid grid_range');
        finite(layer.basis_sigma, 'basis_sigma');
        requireValue(layer.basis_sigma > 0, 'basis_sigma must be positive');
        finite(layer.scale_base, 'scale_base');
        finite(layer.scale_rbf, 'scale_rbf');
        matrix(layer.base_weights, layer.output_features, layer.input_features, 'base_weights');
        if (layer.rbf_scalers !== null) {
            matrix(layer.rbf_scalers, layer.output_features, layer.input_features, 'rbf_scalers');
        }
        requireValue(Array.isArray(layer.rbf_coefficients) &&
            layer.rbf_coefficients.length === layer.output_features, 'Invalid rbf_coefficients');
        layer.rbf_coefficients.forEach(row => {
            requireValue(Array.isArray(row) && row.length === layer.input_features, 'Invalid rbf_coefficients inputs');
            row.forEach(coefficients => {
                requireValue(Array.isArray(coefficients) && coefficients.length === layer.grid_points.length,
                    'RBF coefficients must match the Gaussian centers');
                coefficients.forEach(x => finite(x, 'rbf_coefficients'));
            });
        });
    }

    // erf(x) = 2*x*exp(-x*x)/sqrt(pi) * sum (2*x*x)^n / (2*n+1)!!.
    // The positive-term series avoids cancellation and converges to double precision.
    function erf(x) {
        const a = Math.abs(x);
        if (a >= 6) return Math.sign(x);
        let term = 1;
        let sum = 1;
        for (let n = 1; n < 200; n++) {
            term *= 2 * a * a / (2 * n + 1);
            sum += term;
            if (term <= sum * Number.EPSILON) break;
        }
        return Math.sign(x) * 2 / Math.sqrt(Math.PI) * a * Math.exp(-a * a) * sum;
    }

    function activate(name, x) {
        switch (name) {
            case 'silu': return x / (1 + Math.exp(-x));
            case 'relu': return Math.max(0, x);
            case 'gelu': return 0.5 * x * (1 + erf(x / Math.SQRT2));
            case 'tanh': return Math.tanh(x);
            default: throw new Error(`Unsupported activation: ${name}`);
        }
    }

    function evaluateEdge(layer, inIdx, outIdx, x) {
        const clamped = Math.max(layer.grid_range[0], Math.min(layer.grid_range[1], x));
        const coefficients = layer.rbf_coefficients[outIdx][inIdx];
        let rbf = 0;
        for (let c = 0; c < layer.grid_points.length; c++) {
            const distance = (clamped - layer.grid_points[c]) / layer.basis_sigma;
            rbf += coefficients[c] * Math.exp(-0.5 * distance * distance);
        }
        const scaler = layer.rbf_scalers === null ? 1 : layer.rbf_scalers[outIdx][inIdx];
        return layer.scale_base * layer.base_weights[outIdx][inIdx] * activate(layer.base_activation, x) +
            layer.scale_rbf * scaler * rbf;
    }

    function edgeValue(layer, inIdx, outIdx, x) {
        validateLayer(layer);
        requireValue(Number.isInteger(inIdx) && inIdx >= 0 && inIdx < layer.input_features &&
            Number.isInteger(outIdx) && outIdx >= 0 && outIdx < layer.output_features, 'Invalid edge indices');
        finite(x, 'edge input');
        return evaluateEdge(layer, inIdx, outIdx, x);
    }

    function forward(model, input) {
        requireValue(model && model.metadata && model.metadata.basis === 'gaussian_rbf',
            'Unsupported model schema: expected gaussian_rbf');
        const architecture = model.metadata.architecture;
        requireValue(Array.isArray(architecture) && architecture.length >= 2 &&
            architecture.every(size => Number.isInteger(size) && size > 0), 'Invalid architecture');
        requireValue(Array.isArray(model.layers) && model.layers.length === architecture.length - 1,
            'Layers do not match the architecture');
        requireValue(Array.isArray(input) && input.length === architecture[0], 'Wrong input dimension');
        input.forEach(x => finite(x, 'input'));
        const activationValues = [input.slice()];
        const edges = [];
        model.layers.forEach((layer, index) => {
            validateLayer(layer);
            requireValue(layer.input_features === architecture[index] &&
                layer.output_features === architecture[index + 1], 'Layer dimensions do not match the architecture');
            const current = activationValues[index];
            const contributions = [];
            const next = [];
            for (let outIdx = 0; outIdx < layer.output_features; outIdx++) {
                const row = [];
                let sum = 0;
                for (let inIdx = 0; inIdx < layer.input_features; inIdx++) {
                    const value = evaluateEdge(layer, inIdx, outIdx, current[inIdx]);
                    finite(value, 'edge contribution');
                    row.push(value);
                    sum += value;
                }
                finite(sum, 'layer output');
                contributions.push(row);
                next.push(sum);
            }
            edges.push(contributions);
            activationValues.push(next);
        });
        return { output: activationValues[activationValues.length - 1], activations: activationValues, edges };
    }

    function targetValue(targetId, input) {
        const dimensions = { '1d_sine_wave': 1, '2d_gaussian': 2, '2d_complex': 2 };
        requireValue(Object.prototype.hasOwnProperty.call(dimensions, targetId), `Unknown target: ${targetId}`);
        requireValue(Array.isArray(input) && input.length === dimensions[targetId], 'Wrong target input dimension');
        input.forEach(x => finite(x, 'target input'));
        const [x, y] = input;
        switch (targetId) {
            case '1d_sine_wave': return Math.sin(3 * x) + 0.3 * Math.cos(10 * x);
            case '2d_gaussian': return Math.sin(x) * Math.exp(-y * y);
            case '2d_complex': return Math.sin(x * y) + 0.5 * Math.tanh(x - y);
        }
    }

    return { forward, edgeValue, targetValue };
});
