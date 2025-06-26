// training progress visualization module
class TrainingVisualization {
    constructor() {
        this.model = null;
    }
    
    render(model) {
        console.log('rendering training visualization...');
        
        this.model = model;
        
        if (model.training_history) {
            this.plotTrainingCurves();
            this.displayTrainingStats();
        } else {
            this.showNoDataMessage();
        }
        
        console.log('training visualization ready');
    }
    
    plotTrainingCurves() {
        const history = this.model.training_history;
        
        if (!history || !history.loss) {
            this.showNoDataMessage();
            return;
        }
        
        // create loss curve
        const lossTrace = {
            x: Array.from({length: history.loss.length}, (_, i) => i + 1),
            y: history.loss,
            type: 'scatter',
            mode: 'lines',
            name: 'training loss',
            line: {
                color: '#ff6b6b',
                width: 3
            }
        };
        
        const traces = [lossTrace];
        
        // add validation loss if available
        if (history.val_loss) {
            const valLossTrace = {
                x: Array.from({length: history.val_loss.length}, (_, i) => i + 1),
                y: history.val_loss,
                type: 'scatter',
                mode: 'lines',
                name: 'validation loss',
                line: {
                    color: '#667eea',
                    width: 3,
                    dash: 'dash'
                }
            };
            traces.push(valLossTrace);
        }
        
        // create learning rate subplot if available
        let subplots = [];
        if (history.learning_rate) {
            const lrTrace = {
                x: Array.from({length: history.learning_rate.length}, (_, i) => i + 1),
                y: history.learning_rate,
                type: 'scatter',
                mode: 'lines',
                name: 'learning rate',
                line: {
                    color: '#4ecdc4',
                    width: 2
                },
                yaxis: 'y2'
            };
            traces.push(lrTrace);
        }
        
        const layout = {
            title: 'training progress',
            xaxis: {
                title: 'epoch',
                gridcolor: '#eee'
            },
            yaxis: {
                title: 'loss',
                type: 'log',
                gridcolor: '#eee'
            },
            yaxis2: history.learning_rate ? {
                title: 'learning rate',
                overlaying: 'y',
                side: 'right',
                type: 'log'
            } : undefined,
            legend: {
                x: 0.02,
                y: 0.98,
                bgcolor: 'rgba(255,255,255,0.8)'
            },
            plot_bgcolor: '#fafafa',
            paper_bgcolor: 'white',
            margin: { t: 60, r: 80, b: 60, l: 80 },
            height: 400
        };
        
        // create convergence analysis subplot
        this.plotConvergenceAnalysis(history);
        
        const config = {
            displayModeBar: true,
            modeBarButtonsToRemove: ['pan2d', 'lasso2d', 'select2d'],
            responsive: true
        };
        
        Plotly.newPlot('training-plots', traces, layout, config);
    }
    
    plotConvergenceAnalysis(history) {
        // create a second plot for convergence analysis
        const plotsContainer = document.getElementById('training-plots');
        
        // create convergence rate plot
        const convergenceDiv = document.createElement('div');
        convergenceDiv.id = 'convergence-plot';
        convergenceDiv.style.height = '300px';
        convergenceDiv.style.marginTop = '20px';
        plotsContainer.appendChild(convergenceDiv);
        
        // compute convergence metrics
        const windowSize = Math.min(10, Math.floor(history.loss.length / 10));
        const convergenceRate = this.computeConvergenceRate(history.loss, windowSize);
        
        const convergenceTrace = {
            x: Array.from({length: convergenceRate.length}, (_, i) => i + windowSize),
            y: convergenceRate,
            type: 'scatter',
            mode: 'lines+markers',
            name: 'convergence rate',
            line: {
                color: '#95a5a6',
                width: 2
            },
            marker: {
                size: 4
            }
        };
        
        // add gradient norm if available
        const traces = [convergenceTrace];
        if (history.grad_norm) {
            const gradTrace = {
                x: Array.from({length: history.grad_norm.length}, (_, i) => i + 1),
                y: history.grad_norm,
                type: 'scatter',
                mode: 'lines',
                name: 'gradient norm',
                line: {
                    color: '#e74c3c',
                    width: 2
                },
                yaxis: 'y2'
            };
            traces.push(gradTrace);
        }
        
        const convergenceLayout = {
            title: 'convergence analysis',
            xaxis: {
                title: 'epoch',
                gridcolor: '#eee'
            },
            yaxis: {
                title: 'convergence rate',
                gridcolor: '#eee'
            },
            yaxis2: history.grad_norm ? {
                title: 'gradient norm',
                overlaying: 'y',
                side: 'right',
                type: 'log'
            } : undefined,
            legend: {
                x: 0.02,
                y: 0.98,
                bgcolor: 'rgba(255,255,255,0.8)'
            },
            plot_bgcolor: '#fafafa',
            paper_bgcolor: 'white',
            margin: { t: 60, r: 80, b: 60, l: 80 },
            height: 300
        };
        
        Plotly.newPlot('convergence-plot', traces, convergenceLayout, {
            displayModeBar: false,
            responsive: true
        });
    }
    
    computeConvergenceRate(losses, windowSize) {
        const rates = [];
        
        for (let i = windowSize; i < losses.length; i++) {
            const currentWindow = losses.slice(i - windowSize, i);
            const previousWindow = losses.slice(i - windowSize - 1, i - 1);
            
            const currentAvg = currentWindow.reduce((a, b) => a + b) / currentWindow.length;
            const previousAvg = previousWindow.reduce((a, b) => a + b) / previousWindow.length;
            
            const rate = (previousAvg - currentAvg) / previousAvg;
            rates.push(Math.max(0, rate)); // only positive convergence
        }
        
        return rates;
    }
    
    displayTrainingStats() {
        const history = this.model.training_history;
        const metadata = this.model.metadata;
        const statsContainer = document.getElementById('stats-content');
        
        // compute training statistics
        const finalLoss = history.loss[history.loss.length - 1];
        const initialLoss = history.loss[0];
        const improvementRatio = initialLoss / finalLoss;
        const totalEpochs = history.loss.length;
        
        // find best epoch
        const bestEpoch = history.loss.indexOf(Math.min(...history.loss)) + 1;
        const bestLoss = Math.min(...history.loss);
        
        // compute convergence info
        const convergenceThreshold = initialLoss * 0.01; // 1% of initial loss
        const convergedEpoch = history.loss.findIndex(loss => loss <= convergenceThreshold);
        
        // training speed
        const avgLossReduction = (initialLoss - finalLoss) / totalEpochs;
        
        statsContainer.innerHTML = `
            <div class="stat-item">
                <span>total epochs:</span>
                <strong>${totalEpochs}</strong>
            </div>
            <div class="stat-item">
                <span>final loss:</span>
                <strong>${finalLoss.toExponential(3)}</strong>
            </div>
            <div class="stat-item">
                <span>best loss:</span>
                <strong>${bestLoss.toExponential(3)} (epoch ${bestEpoch})</strong>
            </div>
            <div class="stat-item">
                <span>improvement:</span>
                <strong>${improvementRatio.toFixed(1)}× reduction</strong>
            </div>
            <div class="stat-item">
                <span>convergence:</span>
                <strong>${convergedEpoch > 0 ? `epoch ${convergedEpoch}` : 'not achieved'}</strong>
            </div>
            <div class="stat-item">
                <span>avg reduction/epoch:</span>
                <strong>${avgLossReduction.toExponential(3)}</strong>
            </div>
        `;
        
        // add optimizer info if available
        if (history.optimizer_info) {
            statsContainer.innerHTML += `
                <div class="stat-item">
                    <span>optimizer:</span>
                    <strong>${history.optimizer_info.name || 'unknown'}</strong>
                </div>
                <div class="stat-item">
                    <span>learning rate:</span>
                    <strong>${history.optimizer_info.lr || 'unknown'}</strong>
                </div>
            `;
        }
        
        // add model complexity metrics
        statsContainer.innerHTML += `
            <div style="margin-top: 20px; padding: 15px; background: #f0f8ff; border-radius: 8px;">
                <strong>model complexity:</strong><br>
                • ${metadata.total_parameters} total parameters<br>
                • ${metadata.num_layers} layers<br>
                • grid size: ${metadata.grid_size}<br>
                • spline order: ${metadata.spline_order}
            </div>
        `;
        
        // add training insights
        const insights = this.generateTrainingInsights(history, metadata);
        statsContainer.innerHTML += `
            <div style="margin-top: 20px; padding: 15px; background: #f0fff0; border-radius: 8px;">
                <strong>training insights:</strong><br>
                ${insights}
            </div>
        `;
    }
    
    generateTrainingInsights(history, metadata) {
        const finalLoss = history.loss[history.loss.length - 1];
        const initialLoss = history.loss[0];
        const improvementRatio = initialLoss / finalLoss;
        
        let insights = '';
        
        // convergence assessment
        if (improvementRatio > 1000) {
            insights += '• <strong>excellent convergence</strong> - model learned the function very well<br>';
        } else if (improvementRatio > 100) {
            insights += '• <strong>good convergence</strong> - significant learning achieved<br>';
        } else if (improvementRatio > 10) {
            insights += '• <strong>moderate convergence</strong> - some learning but room for improvement<br>';
        } else {
            insights += '• <strong>limited convergence</strong> - may need more training or tuning<br>';
        }
        
        // loss stability
        const lastTenLosses = history.loss.slice(-10);
        const lossVariability = this.computeVariability(lastTenLosses);
        
        if (lossVariability < 0.01) {
            insights += '• training appears <strong>stable</strong> - loss is converging smoothly<br>';
        } else if (lossVariability < 0.1) {
            insights += '• training shows <strong>minor fluctuations</strong> - generally stable<br>';
        } else {
            insights += '• training shows <strong>instability</strong> - consider reducing learning rate<br>';
        }
        
        // model complexity vs performance
        const paramsPerLayer = metadata.total_parameters / metadata.num_layers;
        if (finalLoss < 1e-4 && paramsPerLayer > 100) {
            insights += '• model may be <strong>overparameterized</strong> for this task<br>';
        } else if (finalLoss > 1e-2 && paramsPerLayer < 50) {
            insights += '• model may benefit from <strong>increased capacity</strong><br>';
        }
        
        return insights;
    }
    
    computeVariability(values) {
        const mean = values.reduce((a, b) => a + b) / values.length;
        const variance = values.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / values.length;
        return Math.sqrt(variance) / mean; // coefficient of variation
    }
    
    showNoDataMessage() {
        const plotsContainer = document.getElementById('training-plots');
        const statsContainer = document.getElementById('stats-content');
        
        plotsContainer.innerHTML = `
            <div style="display: flex; align-items: center; justify-content: center; height: 400px; color: #666;">
                <div style="text-align: center;">
                    <h3>📊 no training data available</h3>
                    <p>this model was loaded without training history.</p>
                    <p>train a new model to see learning curves and statistics.</p>
                </div>
            </div>
        `;
        
        statsContainer.innerHTML = `
            <div class="stat-item">
                <span>status:</span>
                <strong>no data</strong>
            </div>
            <div style="margin-top: 20px; padding: 15px; background: #fff3cd; border-radius: 8px; color: #856404;">
                <strong>note:</strong> training history is only available for models 
                that were trained with the export script. pre-trained models may 
                not include this information.
            </div>
        `;
    }
} 