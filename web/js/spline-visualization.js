// spline function visualization module
class SplineVisualization {
    constructor() {
        this.model = null;
        this.selectedLayer = 0;
        this.selectedConnection = '0-0';
    }
    
    render(model) {
        console.log('rendering spline visualization...');
        
        this.model = model;
        
        // set default selections
        this.selectedLayer = 0;
        this.selectedConnection = '0-0';
        
        // populate selects
        this.populateLayerSelect();
        this.populateConnectionSelect();
        
        // render initial spline
        this.renderSpline();
        
        console.log('spline visualization ready');
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
        
        // add event listener
        select.addEventListener('change', (e) => {
            this.selectLayer(parseInt(e.target.value));
        });
    }
    
    populateConnectionSelect() {
        const select = document.getElementById('connection-select');
        select.innerHTML = '';
        
        if (this.selectedLayer >= 0 && this.selectedLayer < this.model.layers.length) {
            const layer = this.model.layers[this.selectedLayer];
            
            for (let i = 0; i < layer.input_features; i++) {
                for (let j = 0; j < layer.output_features; j++) {
                    const option = document.createElement('option');
                    option.value = `${i}-${j}`;
                    option.textContent = `input ${i} → output ${j}`;
                    select.appendChild(option);
                }
            }
        }
        
        // add event listener
        select.addEventListener('change', (e) => {
            this.selectConnection(e.target.value);
        });
    }
    
    selectLayer(layerIdx) {
        this.selectedLayer = layerIdx;
        this.populateConnectionSelect();
        this.selectedConnection = '0-0';
        this.renderSpline();
    }
    
    selectConnection(connection) {
        this.selectedConnection = connection;
        this.renderSpline();
    }
    
    renderSpline() {
        const layer = this.model.layers[this.selectedLayer];
        const [inputIdx, outputIdx] = this.selectedConnection.split('-').map(x => parseInt(x));
        
        // get spline data
        const splineData = this.getSplineData(layer, inputIdx, outputIdx);
        
        // plot spline function
        this.plotSplineFunction(splineData);
        
        // update info panel
        this.updateSplineInfo(layer, inputIdx, outputIdx, splineData);
    }
    
    getSplineData(layer, inputIdx, outputIdx) {
        // get spline evaluation from exported data
        const splineEval = layer.spline_evaluations.find(
            evaluation => evaluation.input_idx === inputIdx && evaluation.output_idx === outputIdx
        );
        
        if (splineEval) {
            return {
                x_values: splineEval.x_values,
                y_values: splineEval.y_values,
                grid_points: layer.grid_points,
                coefficients: layer.spline_coefficients[outputIdx][inputIdx],
                base_weight: layer.base_weights[outputIdx][inputIdx],
                scale_base: layer.scale_base,
                scale_spline: layer.scale_spline
            };
        }
        
        // fallback: generate synthetic data
        return this.generateSyntheticSpline(layer, inputIdx, outputIdx);
    }
    
    generateSyntheticSpline(layer, inputIdx, outputIdx) {
        const x_values = [];
        const y_values = [];
        
        for (let i = 0; i < 100; i++) {
            const x = -2 + (4 * i / 99);
            // simple polynomial approximation
            const y = layer.spline_coefficients[outputIdx][inputIdx].reduce((sum, coeff, idx) => {
                return sum + coeff * Math.pow(x, idx);
            }, 0);
            
            x_values.push(x);
            y_values.push(y);
        }
        
        return {
            x_values,
            y_values,
            grid_points: layer.grid_points,
            coefficients: layer.spline_coefficients[outputIdx][inputIdx],
            base_weight: layer.base_weights[outputIdx][inputIdx],
            scale_base: layer.scale_base,
            scale_spline: layer.scale_spline
        };
    }
    
    plotSplineFunction(data) {
        // create base spline trace
        const splineTrace = {
            x: data.x_values,
            y: data.y_values.map(y => y * data.scale_spline),
            type: 'scatter',
            mode: 'lines',
            name: 'spline function',
            line: {
                color: '#667eea',
                width: 3
            }
        };
        
        // create linear base trace
        const linearTrace = {
            x: data.x_values,
            y: data.x_values.map(x => x * data.base_weight * data.scale_base),
            type: 'scatter',
            mode: 'lines',
            name: 'linear base',
            line: {
                color: '#ff6b6b',
                width: 2,
                dash: 'dash'
            }
        };
        
        // create combined function trace
        const combinedTrace = {
            x: data.x_values,
            y: data.x_values.map((x, i) => {
                const splineValue = data.y_values[i] * data.scale_spline;
                const baseValue = x * data.base_weight * data.scale_base;
                return splineValue + baseValue;
            }),
            type: 'scatter',
            mode: 'lines',
            name: 'combined function',
            line: {
                color: '#4ecdc4',
                width: 4
            }
        };
        
        // add grid points
        const gridTrace = {
            x: data.grid_points,
            y: data.grid_points.map(() => 0),
            type: 'scatter',
            mode: 'markers',
            name: 'grid points',
            marker: {
                color: '#333',
                size: 8,
                symbol: 'diamond'
            }
        };
        
        const layout = {
            title: `spline function: layer ${this.selectedLayer + 1}, connection ${this.selectedConnection}`,
            xaxis: {
                title: 'input value',
                range: [-2.2, 2.2],
                gridcolor: '#eee',
                zerolinecolor: '#ccc'
            },
            yaxis: {
                title: 'output value',
                gridcolor: '#eee',
                zerolinecolor: '#ccc'
            },
            legend: {
                x: 0.02,
                y: 0.98,
                bgcolor: 'rgba(255,255,255,0.8)'
            },
            plot_bgcolor: '#fafafa',
            paper_bgcolor: 'white',
            margin: { t: 60, r: 30, b: 60, l: 60 },
            height: 500
        };
        
        const config = {
            displayModeBar: true,
            modeBarButtonsToRemove: ['pan2d', 'lasso2d', 'select2d'],
            responsive: true
        };
        
        Plotly.newPlot('spline-plot', [splineTrace, linearTrace, combinedTrace, gridTrace], layout, config);
    }
    
    updateSplineInfo(layer, inputIdx, outputIdx, data) {
        const equationDiv = document.getElementById('function-equation');
        const statsDiv = document.getElementById('function-stats');
        
        // create equation display
        const baseWeight = data.base_weight.toFixed(4);
        const scaleBase = data.scale_base.toFixed(4);
        const scaleSpline = data.scale_spline.toFixed(4);
        
        equationDiv.innerHTML = `
            <strong>function composition:</strong><br>
            f(x) = scale_base × (x × ${baseWeight}) + scale_spline × spline(x)<br>
            <br>
            where:<br>
            • scale_base = ${scaleBase}<br>
            • scale_spline = ${scaleSpline}<br>
            • spline coefficients: [${data.coefficients.map(c => c.toFixed(3)).join(', ')}]
        `;
        
        // compute statistics
        const splineRange = Math.max(...data.y_values) - Math.min(...data.y_values);
        const splineNorm = Math.sqrt(data.coefficients.reduce((sum, c) => sum + c*c, 0));
        const totalWeight = Math.abs(data.base_weight) + splineNorm;
        
        statsDiv.innerHTML = `
            <div class="stat-item">
                <span>spline complexity:</span>
                <strong>${splineNorm.toFixed(4)}</strong>
            </div>
            <div class="stat-item">
                <span>base weight magnitude:</span>
                <strong>${Math.abs(data.base_weight).toFixed(4)}</strong>
            </div>
            <div class="stat-item">
                <span>function range:</span>
                <strong>±${splineRange.toFixed(3)}</strong>
            </div>
            <div class="stat-item">
                <span>total weight:</span>
                <strong>${totalWeight.toFixed(4)}</strong>
            </div>
            <div class="stat-item">
                <span>grid size:</span>
                <strong>${data.grid_points.length} points</strong>
            </div>
            <div class="stat-item">
                <span>spline/base ratio:</span>
                <strong>${(data.scale_spline / data.scale_base).toFixed(2)}</strong>
            </div>
        `;
        
        // add interpretation
        const interpretation = this.interpretSplineFunction(data);
        statsDiv.innerHTML += `
            <div style="margin-top: 20px; padding: 15px; background: #e8f4fd; border-radius: 8px;">
                <strong>interpretation:</strong><br>
                ${interpretation}
            </div>
        `;
    }
    
    interpretSplineFunction(data) {
        const splineNorm = Math.sqrt(data.coefficients.reduce((sum, c) => sum + c*c, 0));
        const baseWeight = Math.abs(data.base_weight);
        const ratio = data.scale_spline / data.scale_base;
        
        let interpretation = '';
        
        if (ratio > 2) {
            interpretation += 'this connection is <strong>spline-dominated</strong> - the learned function is highly nonlinear. ';
        } else if (ratio < 0.5) {
            interpretation += 'this connection is <strong>linear-dominated</strong> - the base linear transformation is primary. ';
        } else {
            interpretation += 'this connection has <strong>balanced</strong> linear and spline components. ';
        }
        
        if (splineNorm > 1) {
            interpretation += 'the spline function shows complex patterns with high variation.';
        } else if (splineNorm < 0.1) {
            interpretation += 'the spline function is relatively simple and smooth.';
        } else {
            interpretation += 'the spline function has moderate complexity.';
        }
        
        return interpretation;
    }
} 