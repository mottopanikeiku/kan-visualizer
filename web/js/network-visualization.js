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
        const splineNorm = Math.sqrt(splineCoeffs.reduce((sum, c) => sum + c*c, 0));
        return Math.abs(baseWeight) + splineNorm;
    }
    
    drawNetwork() {
        // create edge lines
        const edges = this.svg.selectAll('.edge')
            .data(this.edges)
            .enter()
            .append('line')
            .attr('class', 'edge')
            .attr('x1', d => d.source.x)
            .attr('y1', d => d.source.y)
            .attr('x2', d => d.target.x)
            .attr('y2', d => d.target.y)
            .attr('stroke-width', d => Math.min(5, Math.max(1, d.weight * 2)))
            .attr('opacity', d => Math.min(1, Math.max(0.3, d.weight / 2)))
            .on('click', (event, d) => this.selectEdge(d))
            .on('mouseover', (event, d) => this.highlightEdge(d))
            .on('mouseout', () => this.unhighlightEdges());
        
        // create nodes
        const nodeGroups = this.svg.selectAll('.node-group')
            .data(this.nodes)
            .enter()
            .append('g')
            .attr('class', 'node-group')
            .attr('transform', d => `translate(${d.x}, ${d.y})`);
        
        // node circles
        nodeGroups.append('circle')
            .attr('class', d => `node node-${d.type}`)
            .attr('r', d => d.radius)
            .attr('stroke-width', 2)
            .on('click', (event, d) => this.selectNode(d))
            .on('mouseover', (event, d) => this.highlightNode(d))
            .on('mouseout', () => this.unhighlightNodes());
        
        // node labels
        nodeGroups.append('text')
            .attr('class', 'node-label')
            .attr('dy', 4)
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
            const y = Math.sin(edge.weight * x) * Math.exp(-x*x/4);
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
            // create data particle
            const particle = this.svg.append('circle')
                .attr('class', 'data-particle')
                .attr('r', 4)
                .attr('fill', '#ff6b6b')
                .attr('cx', this.nodes[0].x)
                .attr('cy', this.nodes[0].y);
            
            // animate through layers
            let currentLayer = 0;
            const animateToNextLayer = () => {
                if (currentLayer >= this.nodes.length - 1) {
                    particle.remove();
                    return;
                }
                
                const sourceNodes = this.nodes.filter(n => n.layer === currentLayer);
                const targetNodes = this.nodes.filter(n => n.layer === currentLayer + 1);
                
                if (targetNodes.length > 0) {
                    const targetNode = targetNodes[Math.floor(Math.random() * targetNodes.length)];
                    
                    particle.transition()
                        .duration(1000)
                        .attr('cx', targetNode.x)
                        .attr('cy', targetNode.y)
                        .on('end', () => {
                            currentLayer++;
                            setTimeout(animateToNextLayer, 200);
                        });
                }
            };
            
            animateToNextLayer();
        };
        
        // start animation loop
        this.animation = setInterval(animateDataFlow, 3000);
        animateDataFlow(); // start immediately
    }
    
    stopAnimation() {
        if (this.animation) {
            clearInterval(this.animation);
            this.animation = null;
        }
        
        // remove any existing particles
        this.svg.selectAll('.data-particle').remove();
        
        console.log('network animation stopped');
    }
} 