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
        this.stopAnimation();
        console.log('rendering inference engine...');
        
        this.model = model;
        this.dataset = dataset;
        
        // setup input controls
        this.setupInputControls();
        
        // setup inference network
        this.setupInferenceNetwork();
        this.setupNumericalTables();
        
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
            label.htmlFor = `input-${i}`;
            
            const slider = document.createElement('input');
            slider.id = `input-${i}`;
            slider.setAttribute('aria-describedby', 'view-explanation');
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
        container.innerHTML = '<svg id="inference-svg" viewBox="0 0 400 300" role="img" aria-label="Actual activations and signed edge contributions" aria-describedby="inference-legend"></svg>';
        
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
                    .datum({ layer: layerIdx, index: nodeIdx })
                    .attr('cx', x)
                    .attr('cy', y)
                    .attr('r', 8)
                    .attr('fill', this.getNodeColor(layerIdx, layerCount))
                    .attr('stroke', '#333')
                    .attr('stroke-width', 1)
                    .append('title');
                
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
                        .datum({ layer: layerIdx, input: i, output: j })
                        .attr('x1', x1)
                        .attr('y1', y1)
                        .attr('x2', x2)
                        .attr('y2', y2)
                        .attr('stroke', '#ddd')
                        .attr('stroke-width', 1)
                        .attr('opacity', 0.6)
                        .append('title');
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
        document.getElementById('current-input').textContent =
            `[${this.currentInput.map(x => x.toFixed(1)).join(', ')}]`;
        this.evaluation = this.forwardPass(this.currentInput);
        this.currentOutput = this.evaluation.output[0];
        document.getElementById('current-output').textContent =
            this.evaluation.output.map(value => value.toFixed(3)).join(', ');
        this.targetOutput = this.computeTargetOutput();
        document.getElementById('target-output').textContent =
            `target (${this.model.metadata.target_id}): ${this.targetOutput.toFixed(3)}`;
        this.updateNetworkActivations();
        this.updateNumericalTables();
        this.updateActivationFlow();
    }

    setupNumericalTables() {
        const activationBody = document.getElementById('activation-values');
        const edgeBody = document.getElementById('edge-values');
        activationBody.replaceChildren();
        edgeBody.replaceChildren();
        const appendRow = (body, label) => {
            const row = document.createElement('tr');
            const heading = document.createElement('th');
            heading.scope = 'row';
            heading.textContent = label;
            const value = document.createElement('td');
            row.append(heading, value);
            body.appendChild(row);
            return value;
        };
        this.activationCells = this.model.metadata.architecture.map((count, layer) =>
            Array.from({ length: count }, (_, node) => appendRow(
                activationBody, `${layer === 0 ? 'Input' : `Layer ${layer}`}, node ${node}`
            ))
        );
        this.edgeCells = this.model.layers.map((layer, index) =>
            Array.from({ length: layer.output_features }, (_, output) =>
                Array.from({ length: layer.input_features }, (_, input) => appendRow(
                    edgeBody, `Layer ${index + 1}, input ${input}, output ${output}`
                ))
            )
        );
    }

    updateNumericalTables() {
        this.evaluation.activations.forEach((values, layer) =>
            values.forEach((value, node) => {
                this.activationCells[layer][node].textContent = value.toFixed(6);
            })
        );
        this.evaluation.edges.forEach((outputs, layer) =>
            outputs.forEach((inputs, output) =>
                inputs.forEach((value, input) => {
                    this.edgeCells[layer][output][input].textContent = value.toFixed(6);
                })
            )
        );
    }

    forwardPass(input) {
        return KANForward.forward(this.model, input);
    }

    computeTargetOutput() {
        return KANForward.targetValue(this.model.metadata.target_id, this.currentInput);
    }

    updateNetworkActivations() {
        this.evaluation.activations.forEach((values, layer) => {
            values.forEach((value, index) => {
                d3.select(`#inference-svg .node-${layer}-${index}`).text(value.toFixed(3));
            });
        });
        const magnitude = Math.max(...this.evaluation.activations.flat().map(Math.abs), 1e-12);
        const nodes = d3.select('#inference-svg').selectAll('.inference-node');
        nodes.attr('r', node => 6 + 4 * Math.abs(this.evaluation.activations[node.layer][node.index]) / magnitude);
        nodes.select('title').text(node =>
            `layer ${node.layer + 1}, node ${node.index}: activation ${this.evaluation.activations[node.layer][node.index].toFixed(6)}`
        );
        this.highlightActiveConnections();
    }

    highlightActiveConnections() {
        const magnitude = Math.max(...this.evaluation.edges.flat(2).map(Math.abs), 1e-12);
        const value = edge => this.evaluation.edges[edge.layer][edge.output][edge.input];
        const edges = d3.select('#inference-svg').selectAll('.inference-edge');
        edges.attr('opacity', edge => 0.1 + 0.9 * Math.abs(value(edge)) / magnitude)
            .attr('stroke-width', edge => 1 + 3 * Math.abs(value(edge)) / magnitude)
            .attr('stroke', edge => value(edge) < 0 ? '#ff6b6b' : '#667eea');
        edges.select('title').text(edge =>
            `layer ${edge.layer + 1}, input ${edge.input} → output ${edge.output}: full contribution ${value(edge).toFixed(6)}`
        );
    }

    updateActivationFlow() {
        const plotData = {
            x: this.evaluation.activations.map((_, layer) => layer === 0 ? 'input' : `layer ${layer}`),
            y: this.evaluation.activations.map(values =>
                values.reduce((sum, value) => sum + Math.abs(value), 0) / values.length),
            type: 'scatter',
            mode: 'lines+markers',
            line: { color: '#667eea', width: 3 },
            marker: { size: 8, color: '#ff6b6b' }
        };
        Plotly.newPlot('activation-plot', [plotData], {
            title: 'mean |activation|',
            xaxis: { title: 'input / layer' },
            yaxis: { title: 'mean |activation|' },
            margin: { t: 40, r: 20, b: 40, l: 50 },
            height: 200,
            showlegend: false
        }, { displayModeBar: false, responsive: true });
    }
    
    startAnimation() {
        console.log('starting inference animation...');
        this.stopAnimation();
        let tick = 0;
        
        // animate input sliders automatically
        const sliders = document.querySelectorAll('#input-sliders input[type="range"]');
        
        this.animationInterval = setInterval(() => {
            tick++;
            sliders.forEach((slider, i) => {
                const newValue = Math.round(Math.sin(tick / 10 + i) * 15) / 10;
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