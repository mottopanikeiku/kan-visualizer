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
        this.stopAnimation();
        this.model = model;
        this.selectedEdge = null;
        document.getElementById('network-selection').textContent = '';
        document.getElementById('edge-details').innerHTML = '<h3>edge function</h3><p>select an edge to see its full sampled function</p>';
        document.getElementById('layer-info').textContent = 'select a node to see details';

        // clear previous
        d3.select('#network-svg').selectAll('*').remove();

        // setup svg
        this.svg = d3.select('#network-svg');
        const rect = this.svg.node().getBoundingClientRect();
        this.width = rect.width;
        this.height = rect.height;
        this.svg.attr('viewBox', `0 0 ${this.width} ${this.height}`);

        // create network layout
        this.createNetworkLayout(model);

        // draw network
        this.drawNetwork();
        this.setupSelectionControls();

        console.log('network visualization complete');
    }
    setupSelectionControls() {
        const nodeSelect = document.getElementById('network-node-select');
        const edgeSelect = document.getElementById('network-edge-select');
        nodeSelect.replaceChildren(new Option('Choose a node', ''));
        edgeSelect.replaceChildren(new Option('Choose an edge', ''));
        this.nodes.forEach(node => {
            nodeSelect.add(new Option(this.nodeName(node), String(node.id)));
        });
        this.edges.forEach(edge => {
            edgeSelect.add(new Option(`layer ${edge.layerIdx + 1}, input ${edge.sourceIdx} → output ${edge.targetIdx}`, String(edge.id)));
        });
        nodeSelect.onchange = () => {
            if (nodeSelect.value !== '') this.selectNode(this.nodes[Number(nodeSelect.value)]);
        };
        edgeSelect.onchange = () => {
            if (edgeSelect.value !== '') this.selectEdge(this.edges[Number(edgeSelect.value)]);
        };
    }

    // Node layer L is the output of edge layer L, matching the h{L}_i labels and inference tables.
    nodeName(node) {
        return node.layer === 0 ? `input, node ${node.index}` : `${node.type}, layer ${node.layer}, node ${node.index}`;
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
        const data = Utils.edgeSamples(model.layers[layerIdx], sourceIdx, targetIdx);
        return Utils.sampleRms(data.y_values);
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
                    .transition().duration(window.matchMedia('(prefers-reduced-motion: reduce)').matches ? 0 : 200)
                    .attr('r', d.radius * 1.15)
                    .attr('filter', 'url(#glow)');
            })
            .on('mouseout', function (event, d) {
                d3.select(this)
                    .transition().duration(window.matchMedia('(prefers-reduced-motion: reduce)').matches ? 0 : 200)
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
        document.getElementById('network-edge-select').value = String(edge.id);
        document.getElementById('network-selection').textContent =
            `Edge: layer ${edge.layerIdx + 1}, input ${edge.sourceIdx} to output ${edge.targetIdx}. Sampled full-edge RMS ${edge.weight.toFixed(4)}.`;

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
        document.getElementById('network-node-select').value = String(node.id);
        this.svg.selectAll('.node').classed('selected', d => d.id === node.id);
        this.highlightNode(node);
        document.getElementById('network-selection').textContent =
            `Node: ${this.nodeName(node)}.`;
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

        const data = this.getSplineDataForEdge(edge);
        detailsDiv.innerHTML = `
            <h3>Gaussian RBF edge function</h3>
            <p><strong>connection:</strong> layer ${edge.layerIdx + 1}, input ${edge.sourceIdx} → output ${edge.targetIdx}</p>
            <p><strong>sampled full-edge RMS:</strong> ${edge.weight.toFixed(4)}</p>
            <p>I use ${data.y_values.length} samples on [${Math.min(...data.x_values).toFixed(2)}, ${Math.max(...data.x_values).toFixed(2)}], including the scaled base activation and RBF branch. This RMS is not feature importance.</p>
            <div id="edge-spline-plot" role="img" aria-label="Selected full sampled edge function"></div>
        `;
        this.plotEdgeSpline(data, 'edge-spline-plot');
    }

    showNodeDetails(node) {
        const detailsDiv = document.getElementById('layer-info');

        detailsDiv.innerHTML = `
            <h3>node details</h3>
            <p><strong>type:</strong> ${node.type}</p>
            <p><strong>layer:</strong> ${node.layer === 0 ? 'input' : node.layer}</p>
            <p><strong>index:</strong> ${node.index}</p>
        `;
    }

    getSplineDataForEdge(edge) {
        return Utils.edgeSamples(this.model.layers[edge.layerIdx], edge.sourceIdx, edge.targetIdx);
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
            name: 'full edge contribution'
        };

        const layout = {
            margin: { t: 20, r: 20, b: 40, l: 40 },
            xaxis: { title: 'input' },
            yaxis: { title: 'edge contribution' },
            showlegend: false,
            height: 200
        };

        Plotly.newPlot(containerId, [trace], layout, {
            displayModeBar: false,
            responsive: true
        });
    }

    startAnimation() {
        this.stopAnimation();
        let pathIndex = 0;
        const layerCount = this.model.metadata.architecture.length;
        const animatePath = () => {
            // Deterministic illustration of connectivity, not an activation measurement.
            const path = Array.from({ length: layerCount }, (_, layer) => {
                const nodes = this.nodes.filter(node => node.layer === layer);
                return nodes[(pathIndex + layer) % nodes.length];
            });
            pathIndex++;
            const particle = this.svg.append('circle')
                .attr('class', 'data-particle')
                .attr('r', 5)
                .attr('fill', '#ff6b6b')
                .attr('cx', path[0].x)
                .attr('cy', path[0].y);
            let transition = particle.transition();
            for (let layer = 1; layer < path.length; layer++) {
                transition = transition.duration(500).ease(d3.easeCubicInOut)
                    .attr('cx', path[layer].x).attr('cy', path[layer].y).transition();
            }
            transition.duration(200).attr('opacity', 0).remove();
        };
        this.animation = setInterval(animatePath, layerCount * 500 + 500);
        animatePath();
    }

    stopAnimation() {
        if (this.animation) {
            clearInterval(this.animation);
            this.animation = null;
        }

        if (this.svg) {
            this.svg.selectAll('.data-particle').interrupt().remove();
        }

        console.log('network animation stopped');
    }
} 