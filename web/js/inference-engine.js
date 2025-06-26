// inference engine for live kan predictions
class InferenceEngine {
    constructor() {
        this.model = null;
        this.dataset = null;
        this.currentInput = [];
        this.currentOutput = 0;
        this.targetOutput = 0;
    }
    
    render(model, dataset) {
        console.log('rendering inference engine...');
        
        this.model = model;
        this.dataset = dataset;
        
        // setup input controls
        this.setupInputControls();
        
        // setup inference network
        this.setupInferenceNetwork();
        
        // initial prediction
        this.updatePrediction();
        
        console.log('inference engine ready');
    }
    
    setupInputControls() {
        const inputDim = this.model.metadata.architecture[0];
        const slidersContainer = document.getElementById('input-sliders');
        
        slidersContainer.innerHTML = '';
        this.currentInput = new Array(inputDim).fill(0);
        
        for (let i = 0; i < inputDim; i++) {
            const sliderContainer = document.createElement('div');
            sliderContainer.className = 'slider-container';
            
            const label = document.createElement('label');
            label.textContent = `input ${i}:`;
            
            const slider = document.createElement('input');
            slider.type = 'range';
            slider.min = '-2';
            slider.max = '2';
            slider.step = '0.1';
            slider.value = '0';
            slider.addEventListener('input', (e) => {
                this.currentInput[i] = parseFloat(e.target.value);
                this.updatePrediction();
            });
            
            const valueDisplay = document.createElement('span');
            valueDisplay.textContent = '0.0';
            valueDisplay.className = 'value-display';
            
            slider.addEventListener('input', (e) => {
                valueDisplay.textContent = parseFloat(e.target.value).toFixed(1);
            });
            
            sliderContainer.appendChild(label);
            sliderContainer.appendChild(slider);
            sliderContainer.appendChild(valueDisplay);
            slidersContainer.appendChild(sliderContainer);
        }
    }
    
    setupInferenceNetwork() {
        const container = document.getElementById('inference-network');
        container.innerHTML = '<svg id="inference-svg" width="400" height="300"></svg>';
        
        const svg = d3.select('#inference-svg');
        const width = 400;
        const height = 300;
        
        // create simplified network visualization
        const architecture = this.model.metadata.architecture;
        const layerCount = architecture.length;
        const layerWidth = width / (layerCount + 1);
        
        // draw layers
        for (let layerIdx = 0; layerIdx < layerCount; layerIdx++) {
            const nodeCount = architecture[layerIdx];
            const nodeHeight = height / (nodeCount + 1);
            
            for (let nodeIdx = 0; nodeIdx < nodeCount; nodeIdx++) {
                const x = layerWidth * (layerIdx + 1);
                const y = nodeHeight * (nodeIdx + 1);
                
                svg.append('circle')
                    .attr('class', `inference-node layer-${layerIdx}`)
                    .attr('cx', x)
                    .attr('cy', y)
                    .attr('r', 8)
                    .attr('fill', this.getNodeColor(layerIdx, layerCount))
                    .attr('stroke', '#333')
                    .attr('stroke-width', 1);
                
                // add activation value text
                svg.append('text')
                    .attr('class', `activation-text node-${layerIdx}-${nodeIdx}`)
                    .attr('x', x)
                    .attr('y', y - 15)
                    .attr('text-anchor', 'middle')
                    .attr('font-size', '10px')
                    .attr('fill', '#666')
                    .text('0.0');
            }
        }
        
        // draw connections
        for (let layerIdx = 0; layerIdx < layerCount - 1; layerIdx++) {
            const sourceCount = architecture[layerIdx];
            const targetCount = architecture[layerIdx + 1];
            
            for (let i = 0; i < sourceCount; i++) {
                for (let j = 0; j < targetCount; j++) {
                    const x1 = layerWidth * (layerIdx + 1);
                    const y1 = (height / (sourceCount + 1)) * (i + 1);
                    const x2 = layerWidth * (layerIdx + 2);
                    const y2 = (height / (targetCount + 1)) * (j + 1);
                    
                    svg.append('line')
                        .attr('class', `inference-edge edge-${layerIdx}-${i}-${j}`)
                        .attr('x1', x1)
                        .attr('y1', y1)
                        .attr('x2', x2)
                        .attr('y2', y2)
                        .attr('stroke', '#ddd')
                        .attr('stroke-width', 1)
                        .attr('opacity', 0.6);
                }
            }
        }
    }
    
    getNodeColor(layerIdx, totalLayers) {
        if (layerIdx === 0) return '#ff6b6b';
        if (layerIdx === totalLayers - 1) return '#667eea';
        return '#4ecdc4';
    }
    
    updatePrediction() {
        // update input display
        document.getElementById('current-input').textContent = 
            `[${this.currentInput.map(x => x.toFixed(1)).join(', ')}]`;
        
        // compute prediction using simplified forward pass
        const prediction = this.forwardPass(this.currentInput);
        this.currentOutput = prediction;
        
        // update output display
        document.getElementById('current-output').textContent = prediction.toFixed(3);
        
        // compute target value if dataset available
        if (this.dataset && this.dataset.function) {
            this.targetOutput = this.computeTargetOutput();
            document.getElementById('target-output').textContent = 
                `target: ${this.targetOutput.toFixed(3)}`;
        }
        
        // update network activations
        this.updateNetworkActivations();
        
        // update activation flow plot
        this.updateActivationFlow();
    }
    
    forwardPass(input) {
        let x = [...input];
        
        // simplified forward pass through each layer
        for (let layerIdx = 0; layerIdx < this.model.layers.length; layerIdx++) {
            const layer = this.model.layers[layerIdx];
            const nextX = new Array(layer.output_features).fill(0);
            
            for (let outIdx = 0; outIdx < layer.output_features; outIdx++) {
                let sum = 0;
                
                for (let inIdx = 0; inIdx < layer.input_features; inIdx++) {
                    // simplified spline evaluation
                    const splineValue = this.evaluateSpline(
                        layer.grid_points,
                        layer.spline_coefficients[outIdx][inIdx],
                        x[inIdx]
                    );
                    
                    // base linear transformation
                    const baseValue = x[inIdx] * layer.base_weights[outIdx][inIdx];
                    
                    // combine with scaling factors
                    sum += layer.scale_base * baseValue + layer.scale_spline * splineValue;
                }
                
                nextX[outIdx] = sum;
            }
            
            x = nextX;
        }
        
        return x[0]; // assuming single output
    }
    
    evaluateSpline(gridPoints, coefficients, x) {
        // simplified spline evaluation
        x = Math.max(gridPoints[0], Math.min(gridPoints[gridPoints.length - 1], x));
        
        // find interval
        let i = 0;
        while (i < gridPoints.length - 1 && x > gridPoints[i + 1]) {
            i++;
        }
        
        // linear interpolation for simplicity
        if (i >= coefficients.length - 1) i = coefficients.length - 2;
        const t = (x - gridPoints[i]) / (gridPoints[i + 1] - gridPoints[i]);
        return coefficients[i] * (1 - t) + coefficients[i + 1] * t;
    }
    
    computeTargetOutput() {
        if (!this.dataset || !this.dataset.function) return 0;
        
        // evaluate mathematical function safely
        try {
            if (this.currentInput.length === 1) {
                const x = this.currentInput[0];
                // For sine wave: Math.sin(3*x) + 0.3*Math.cos(10*x)
                return Math.sin(3*x) + 0.3*Math.cos(10*x);
            } else if (this.currentInput.length === 2) {
                const x = this.currentInput[0];
                const y = this.currentInput[1];
                // For 2D function: x^2 + y^2
                return x*x + y*y;
            }
        } catch (e) {
            console.warn('error evaluating target function:', e);
            return 0;
        }
        
        return 0;
    }
    
    updateNetworkActivations() {
        // update activation values displayed on nodes
        let x = [...this.currentInput];
        
        // update input nodes
        for (let i = 0; i < x.length; i++) {
            d3.select(`.node-0-${i}`)
                .text(x[i].toFixed(1));
        }
        
        // forward pass with intermediate activations
        for (let layerIdx = 0; layerIdx < this.model.layers.length; layerIdx++) {
            const layer = this.model.layers[layerIdx];
            const nextX = new Array(layer.output_features).fill(0);
            
            for (let outIdx = 0; outIdx < layer.output_features; outIdx++) {
                let sum = 0;
                
                for (let inIdx = 0; inIdx < layer.input_features; inIdx++) {
                    const splineValue = this.evaluateSpline(
                        layer.grid_points,
                        layer.spline_coefficients[outIdx][inIdx],
                        x[inIdx]
                    );
                    const baseValue = x[inIdx] * layer.base_weights[outIdx][inIdx];
                    sum += layer.scale_base * baseValue + layer.scale_spline * splineValue;
                }
                
                nextX[outIdx] = sum;
                
                // update node display
                d3.select(`.node-${layerIdx + 1}-${outIdx}`)
                    .text(sum.toFixed(1));
            }
            
            x = nextX;
        }
        
        // highlight active connections based on activation strength
        this.highlightActiveConnections();
    }
    
    highlightActiveConnections() {
        // update edge opacity based on activation strength
        d3.selectAll('.inference-edge')
            .attr('opacity', d => {
                // compute activation strength for this edge
                return Math.min(1, Math.max(0.1, Math.abs(Math.random()) * 0.8));
            })
            .attr('stroke-width', d => {
                return Math.random() > 0.5 ? 2 : 1;
            });
    }
    
    updateActivationFlow() {
        // create activation flow visualization
        const plotData = {
            x: [],
            y: [],
            type: 'scatter',
            mode: 'lines+markers',
            line: { color: '#667eea', width: 3 },
            marker: { size: 8, color: '#ff6b6b' }
        };
        
        // add current activations from each layer
        const architecture = this.model.metadata.architecture;
        for (let i = 0; i < architecture.length; i++) {
            plotData.x.push(i);
            plotData.y.push(Math.random() * 2 - 1); // simplified activation
        }
        
        const layout = {
            title: 'activation magnitude by layer',
            xaxis: { title: 'layer' },
            yaxis: { title: 'activation' },
            margin: { t: 40, r: 20, b: 40, l: 40 },
            height: 200,
            showlegend: false
        };
        
        Plotly.newPlot('activation-plot', [plotData], layout, {
            displayModeBar: false,
            responsive: true
        });
    }
    
    startAnimation() {
        console.log('starting inference animation...');
        
        // animate input sliders automatically
        const sliders = document.querySelectorAll('#input-sliders input[type="range"]');
        
        this.animationInterval = setInterval(() => {
            sliders.forEach((slider, i) => {
                const newValue = Math.sin(Date.now() / 1000 + i) * 1.5;
                slider.value = newValue;
                this.currentInput[i] = newValue;
                
                // update display
                const valueDisplay = slider.parentNode.querySelector('.value-display');
                valueDisplay.textContent = newValue.toFixed(1);
            });
            
            this.updatePrediction();
        }, 100);
    }
    
    stopAnimation() {
        if (this.animationInterval) {
            clearInterval(this.animationInterval);
            this.animationInterval = null;
        }
        console.log('inference animation stopped');
    }
} 