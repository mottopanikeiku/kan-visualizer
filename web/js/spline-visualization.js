// Gaussian-RBF edge function visualization; historical class/DOM names are internal.
class SplineVisualization {
    constructor() {
        this.model = null;
        this.selectedLayer = 0;
        this.selectedConnection = '0-0';
    }

    render(model) {
        this.model = model;
        this.selectedLayer = 0;
        this.selectedConnection = '0-0';
        this.populateLayerSelect();
        this.populateConnectionSelect();
        this.renderSpline();
    }

    populateLayerSelect() {
        const select = document.getElementById('layer-select');
        select.innerHTML = '';
        this.model.layers.forEach((layer, i) => {
            const option = document.createElement('option');
            option.value = i;
            option.textContent = `layer ${i + 1} (${layer.input_features} → ${layer.output_features})`;
            select.appendChild(option);
        });
        select.onchange = event => this.selectLayer(Number(event.target.value));
    }

    populateConnectionSelect() {
        const select = document.getElementById('connection-select');
        select.innerHTML = '';
        const layer = this.model.layers[this.selectedLayer];
        for (let i = 0; i < layer.input_features; i++) {
            for (let j = 0; j < layer.output_features; j++) {
                const option = document.createElement('option');
                option.value = `${i}-${j}`;
                option.textContent = `input ${i} → output ${j}`;
                select.appendChild(option);
            }
        }
        select.onchange = event => this.selectConnection(event.target.value);
    }

    selectLayer(layerIdx) {
        this.selectedLayer = layerIdx;
        this.selectedConnection = '0-0';
        this.populateConnectionSelect();
        this.renderSpline();
    }

    selectConnection(connection) {
        this.selectedConnection = connection;
        this.renderSpline();
    }

    renderSpline() {
        const layer = this.model.layers[this.selectedLayer];
        const [inputIdx, outputIdx] = this.selectedConnection.split('-').map(Number);
        const data = Utils.edgeSamples(layer, inputIdx, outputIdx);
        this.plotSplineFunction(layer, data);
        this.updateSplineInfo(layer, inputIdx, outputIdx, data);
    }

    plotSplineFunction(layer, data) {
        const edgeTrace = {
            x: data.x_values,
            y: data.y_values,
            type: 'scatter',
            mode: 'lines',
            name: 'full edge contribution (base + Gaussian RBF)',
            line: { color: '#667eea', width: 3 }
        };
        const centerTrace = {
            x: layer.grid_points,
            y: layer.grid_points.map(() => 0),
            type: 'scatter',
            mode: 'markers',
            name: 'Gaussian centers (shown at zero)',
            marker: { color: '#333', size: 8, symbol: 'diamond' }
        };
        const layout = {
            title: `edge function: layer ${this.selectedLayer + 1}, connection ${this.selectedConnection}`,
            xaxis: { title: 'input value', gridcolor: '#eee', zerolinecolor: '#ccc' },
            yaxis: { title: 'full edge contribution', gridcolor: '#eee', zerolinecolor: '#ccc' },
            legend: { x: 0.02, y: 0.98, bgcolor: 'rgba(255,255,255,0.8)' },
            plot_bgcolor: '#fafafa',
            paper_bgcolor: 'white',
            margin: { t: 60, r: 30, b: 60, l: 60 },
            height: 500
        };
        Plotly.newPlot('spline-plot', [edgeTrace, centerTrace], layout, {
            displayModeBar: true,
            modeBarButtonsToRemove: ['pan2d', 'lasso2d', 'select2d'],
            responsive: true
        });
    }

    updateSplineInfo(layer, inputIdx, outputIdx, data) {
        const coefficients = layer.rbf_coefficients[outputIdx][inputIdx];
        const scaler = layer.rbf_scalers === null ? 1 : layer.rbf_scalers[outputIdx][inputIdx];
        document.getElementById('function-equation').innerHTML = `
            <strong>full edge function:</strong><br>
            f(x) = scale_base × base_weight × ${layer.base_activation}(x)<br>
            + scale_rbf × rbf_scaler × Σ cₖ exp(−(clamp(x) − μₖ)² / (2σ²))<br><br>
            base_weight = ${layer.base_weights[outputIdx][inputIdx].toFixed(4)}<br>
            scale_base = ${layer.scale_base.toFixed(4)}<br>
            scale_rbf = ${layer.scale_rbf.toFixed(4)}<br>
            rbf_scaler = ${scaler.toFixed(4)}${layer.rbf_scalers === null ? ' (standalone scaling disabled)' : ''}<br>
            σ = ${layer.basis_sigma.toFixed(4)}<br>
            clamp range = [${layer.grid_range.join(', ')}] (RBF branch only)<br>
            RBF coefficients = [${coefficients.map(c => c.toFixed(3)).join(', ')}]
        `;
        const min = Math.min(...data.y_values);
        const max = Math.max(...data.y_values);
        document.getElementById('function-stats').innerHTML = `
            <div class="stat-item"><span>sampled full-edge RMS:</span><strong>${Utils.sampleRms(data.y_values).toFixed(4)}</strong></div>
            <div class="stat-item"><span>sampled output range:</span><strong>[${min.toFixed(3)}, ${max.toFixed(3)}]</strong></div>
            <div class="stat-item"><span>grid:</span><strong>${layer.grid_points.length - 1} intervals / ${layer.grid_points.length} centers</strong></div>
            <div class="stat-item"><span>sampled input range:</span><strong>[${Math.min(...data.x_values).toFixed(2)}, ${Math.max(...data.x_values).toFixed(2)}]</strong></div>
            <p>statistics use ${data.y_values.length} exported full-edge samples, not a measure of feature importance. the base activation uses the unclamped input.</p>
        `;
    }
}
