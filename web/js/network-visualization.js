// network visualization module
class NetworkVisualization {
    constructor() {
        this.svg = null;
        this.width = 0;
        this.height = 0;
        this.nodes = [];
        this.edges = [];
        this.animation = null;
        this.selectedEdge = null;
    }

    render(model) {
        console.log('rendering network visualization...');

        // clear previous
        d3.select('#network-svg').selectAll('*').remove();

        // setup svg
        this.svg = d3.select('#network-svg');
        const rect = this.svg.node().getBoundingClientRect();
        this.width = rect.width;
        this.height = rect.height;

        // create network layout
        this.createNetworkLayout(model);

        // draw network
        this.drawNetwork();

        console.log('network visualization complete');
    }

    createNetworkLayout(model) {
        this.nodes = [];
        this.edges = [];

        const architecture = model.metadata.architecture;
        const layerCount = architecture.length;
        const maxNodesInLayer = Math.max(...architecture);

        // calculate positions
        const layerWidth = this.width / (layerCount + 1);
        const nodeRadius = Math.min(25, this.height / (maxNodesInLayer * 3));

        // create nodes
        let nodeId = 0;
        for (let layerIdx = 0; layerIdx < layerCount; layerIdx++) {
            const nodeCount = architecture[layerIdx];
            const layerHeight = this.height / (nodeCount + 1);

            for (let nodeIdx = 0; nodeIdx < nodeCount; nodeIdx++) {
                this.nodes.push({
                    id: nodeId++,
                    layer: layerIdx,
                    index: nodeIdx,
                    x: layerWidth * (layerIdx + 1),
                    y: layerHeight * (nodeIdx + 1),
                    radius: nodeRadius,
                    type: layerIdx === 0 ? 'input' :
                        layerIdx === layerCount - 1 ? 'output' : 'hidden'
                });
            }
        }

        // create edges
        let edgeId = 0;
        for (let layerIdx = 0; layerIdx < layerCount - 1; layerIdx++) {
            const sourceNodes = this.nodes.filter(n => n.layer === layerIdx);
            const targetNodes = this.nodes.filter(n => n.layer === layerIdx + 1);

            sourceNodes.forEach(source => {
                targetNodes.forEach(target => {
                    this.edges.push({
                        id: edgeId++,
                        source: source,
                        target: target,
                        layerIdx: layerIdx,
                        sourceIdx: source.index,
                        targetIdx: target.index,
                        weight: this.getEdgeWeight(model, layerIdx, source.index, target.index)
                    });
                });
            });
        }
    }

    getEdgeWeight(model, layerIdx, sourceIdx, targetIdx) {
        // get the spline weight magnitude for this connection
        const layer = model.layers[layerIdx];
        const splineCoeffs = layer.spline_coefficients[targetIdx][sourceIdx];
        const baseWeight = layer.base_weights[targetIdx][sourceIdx];

        // combine spline and base weights
        const splineNorm = Math.sqrt(splineCoeffs.reduce((sum, c) => sum + c * c, 0));
        return Math.abs(baseWeight) + splineNorm;
    }

    drawNetwork() {
        // Create gradient definitions for nodes
        const defs = this.svg.append('defs');

        // Input node gradient
        const inputGradient = defs.append('linearGradient')
            .attr('id', 'inputGradient')
            .attr('x1', '0%').attr('y1', '0%')
            .attr('x2', '100%').attr('y2', '100%');
        inputGradient.append('stop').attr('offset', '0%').attr('stop-color', '#ff6b6b');
        inputGradient.append('stop').attr('offset', '100%').attr('stop-color', '#ee5a5a');

        // Hidden node gradient
        const hiddenGradient = defs.append('linearGradient')
            .attr('id', 'hiddenGradient')
            .attr('x1', '0%').attr('y1', '0%')
            .attr('x2', '100%').attr('y2', '100%');
        hiddenGradient.append('stop').attr('offset', '0%').attr('stop-color', '#4ecdc4');
        hiddenGradient.append('stop').attr('offset', '100%').attr('stop-color', '#3db8b0');

        // Output node gradient
        const outputGradient = defs.append('linearGradient')
            .attr('id', 'outputGradient')
            .attr('x1', '0%').attr('y1', '0%')
            .attr('x2', '100%').attr('y2', '100%');
        outputGradient.append('stop').attr('offset', '0%').attr('stop-color', '#667eea');
        outputGradient.append('stop').attr('offset', '100%').attr('stop-color', '#764ba2');

        // Glow filter for hover effects
        const glowFilter = defs.append('filter')
            .attr('id', 'glow')
            .attr('x', '-50%').attr('y', '-50%')
            .attr('width', '200%').attr('height', '200%');
        glowFilter.append('feGaussianBlur').attr('stdDeviation', '3').attr('result', 'coloredBlur');
        const feMerge = glowFilter.append('feMerge');
        feMerge.append('feMergeNode').attr('in', 'coloredBlur');
        feMerge.append('feMergeNode').attr('in', 'SourceGraphic');

        // Calculate weight scale
        const maxWeight = Math.max(...this.edges.map(e => e.weight), 0.1);
        const weightScale = d3.scaleLinear().domain([0, maxWeight]).range([0.3, 1]);
        const widthScale = d3.scaleLinear().domain([0, maxWeight]).range([1, 4]);

        // Create edge lines with improved styling
        const edges = this.svg.selectAll('.edge')
            .data(this.edges)
            .enter()
            .append('line')
            .attr('class', 'edge')
            .attr('x1', d => d.source.x)
            .attr('y1', d => d.source.y)
            .attr('x2', d => d.target.x)
            .attr('y2', d => d.target.y)
            .attr('stroke', '#b0b0b0')
            .attr('stroke-width', d => widthScale(d.weight))
            .attr('opacity', d => weightScale(d.weight))
            .attr('stroke-linecap', 'round')
            .on('click', (event, d) => this.selectEdge(d))
            .on('mouseover', (event, d) => this.highlightEdge(d))
            .on('mouseout', () => this.unhighlightEdges());

        // Create nodes with gradient fills
        const nodeGroups = this.svg.selectAll('.node-group')
            .data(this.nodes)
            .enter()
            .append('g')
            .attr('class', 'node-group')
            .attr('transform', d => `translate(${d.x}, ${d.y})`);

        // Node shadow
        nodeGroups.append('circle')
            .attr('class', 'node-shadow')
            .attr('r', d => d.radius + 2)
            .attr('fill', 'rgba(0,0,0,0.1)')
            .attr('transform', 'translate(2, 2)');

        // Node circles with gradients
        nodeGroups.append('circle')
            .attr('class', d => `node node-${d.type}`)
            .attr('r', d => d.radius)
            .attr('fill', d => {
                if (d.type === 'input') return 'url(#inputGradient)';
                if (d.type === 'output') return 'url(#outputGradient)';
                return 'url(#hiddenGradient)';
            })
            .attr('stroke', d => {
                if (d.type === 'input') return '#e55555';
                if (d.type === 'output') return '#5a67d8';
                return '#45b7b8';
            })
            .attr('stroke-width', 3)
            .style('cursor', 'pointer')
            .on('click', (event, d) => this.selectNode(d))
            .on('mouseover', function (event, d) {
                d3.select(this)
                    .transition().duration(200)
                    .attr('r', d.radius * 1.15)
                    .attr('filter', 'url(#glow)');
            })
            .on('mouseout', function (event, d) {
                d3.select(this)
                    .transition().duration(200)
                    .attr('r', d.radius)
                    .attr('filter', null);
            });

        // Node labels with better styling
        nodeGroups.append('text')
            .attr('class', 'node-label')
            .attr('dy', 4)
            .attr('fill', '#333')
            .attr('font-size', '11px')
            .attr('font-weight', '700')
            .attr('text-anchor', 'middle')
            .attr('pointer-events', 'none')
            .text((d, i) => {
                if (d.type === 'input') return `x${d.index}`;
                if (d.type === 'output') return `y${d.index}`;
                return `h${d.layer}_${d.index}`;
            });
    }

    selectEdge(edge) {
        this.selectedEdge = edge;

        // highlight selected edge
        this.svg.selectAll('.edge').classed('selected', false);
        this.svg.selectAll('.edge')
            .filter(d => d.id === edge.id)
            .classed('selected', true);

        // show edge details
        this.showEdgeDetails(edge);

        console.log('selected edge:', edge);
    }

    selectNode(node) {
        // show node details
        this.showNodeDetails(node);

        console.log('selected node:', node);
    }

    highlightEdge(edge) {
        this.svg.selectAll('.edge')
            .filter(d => d.id === edge.id)
            .classed('highlighted', true);
    }

    unhighlightEdges() {
        this.svg.selectAll('.edge').classed('highlighted', false);
    }

    highlightNode(node) {
        // highlight connected edges
        this.svg.selectAll('.edge')
            .classed('connected', d =>
                d.source.id === node.id || d.target.id === node.id
            );
    }

    unhighlightNodes() {
        this.svg.selectAll('.edge').classed('connected', false);
    }

    showEdgeDetails(edge) {
        const detailsDiv = document.getElementById('edge-details');

        // get spline data for this edge
        const splineData = this.getSplineDataForEdge(edge);

        detailsDiv.innerHTML = `
            <h3>edge function</h3>
            <p><strong>connection:</strong> layer ${edge.layerIdx + 1}, input ${edge.sourceIdx} → output ${edge.targetIdx}</p>
            <p><strong>weight magnitude:</strong> ${edge.weight.toFixed(4)}</p>
            <div id="edge-spline-plot" style="height: 200px; margin-top: 10px;"></div>
        `;

        // plot spline function
        this.plotEdgeSpline(splineData, 'edge-spline-plot');
    }

    showNodeDetails(node) {
        const detailsDiv = document.getElementById('layer-info');

        detailsDiv.innerHTML = `
            <h3>node details</h3>
            <p><strong>type:</strong> ${node.type}</p>
            <p><strong>layer:</strong> ${node.layer + 1}</p>
            <p><strong>index:</strong> ${node.index}</p>
            <p><strong>position:</strong> (${node.x.toFixed(1)}, ${node.y.toFixed(1)})</p>
        `;
    }

    getSplineDataForEdge(edge) {
        // this would get the actual spline evaluation data
        // for now, return mock data
        const x_values = [];
        const y_values = [];

        for (let i = 0; i < 100; i++) {
            const x = -2 + (4 * i / 99);
            const y = Math.sin(edge.weight * x) * Math.exp(-x * x / 4);
            x_values.push(x);
            y_values.push(y);
        }

        return { x_values, y_values };
    }

    plotEdgeSpline(data, containerId) {
        const trace = {
            x: data.x_values,
            y: data.y_values,
            type: 'scatter',
            mode: 'lines',
            line: {
                color: '#667eea',
                width: 3
            },
            name: 'spline function'
        };

        const layout = {
            margin: { t: 20, r: 20, b: 40, l: 40 },
            xaxis: { title: 'input' },
            yaxis: { title: 'output' },
            showlegend: false,
            height: 200
        };

        Plotly.newPlot(containerId, [trace], layout, {
            displayModeBar: false,
            responsive: true
        });
    }

    startAnimation() {
        console.log('starting network animation...');

        const animateDataFlow = () => {
            // Get layers for organized flow
            const architecture = this.nodes.reduce((acc, node) => {
                if (!acc[node.layer]) acc[node.layer] = [];
                acc[node.layer].push(node);
                return acc;
            }, {});

            const layers = Object.keys(architecture).sort((a, b) => a - b).map(k => architecture[k]);

            // Random starting input node
            const startNode = layers[0][Math.floor(Math.random() * layers[0].length)];

            // Create glowing particle group
            const particleGroup = this.svg.append('g').attr('class', 'data-particle-group');

            // Outer glow
            particleGroup.append('circle')
                .attr('class', 'particle-glow')
                .attr('r', 12)
                .attr('fill', 'rgba(255, 107, 107, 0.3)')
                .attr('cx', startNode.x)
                .attr('cy', startNode.y);

            // Core particle
            particleGroup.append('circle')
                .attr('class', 'data-particle')
                .attr('r', 5)
                .attr('fill', '#ff6b6b')
                .attr('cx', startNode.x)
                .attr('cy', startNode.y)
                .style('filter', 'drop-shadow(0 0 6px #ff6b6b)');

            // Animate through all layers
            let currentLayer = 0;

            const animateToNextLayer = () => {
                currentLayer++;

                if (currentLayer >= layers.length) {
                    // Fade out and remove
                    particleGroup.transition()
                        .duration(300)
                        .style('opacity', 0)
                        .remove();
                    return;
                }

                // Pick random target in next layer
                const targetNode = layers[currentLayer][Math.floor(Math.random() * layers[currentLayer].length)];

                // Change color based on layer progression
                const layerColors = ['#ff6b6b', '#4ecdc4', '#667eea', '#764ba2'];
                const newColor = layerColors[Math.min(currentLayer, layerColors.length - 1)];

                // Animate to target
                particleGroup.selectAll('circle')
                    .transition()
                    .duration(500)
                    .ease(d3.easeCubicInOut)
                    .attr('cx', targetNode.x)
                    .attr('cy', targetNode.y)
                    .on('end', function () {
                        // Update particle color on the main particle only
                        d3.select(this.parentNode).select('.data-particle')
                            .attr('fill', newColor)
                            .style('filter', `drop-shadow(0 0 6px ${newColor})`);

                        setTimeout(animateToNextLayer, 150);
                    });
            };

            setTimeout(animateToNextLayer, 200);
        };

        // Start animation loop  
        this.animation = setInterval(animateDataFlow, 2000);
        animateDataFlow(); // start immediately
    }

    stopAnimation() {
        if (this.animation) {
            clearInterval(this.animation);
            this.animation = null;
        }

        // Remove any existing particles
        this.svg.selectAll('.data-particle-group').remove();
        this.svg.selectAll('.data-particle').remove();

        console.log('network animation stopped');
    }
} 