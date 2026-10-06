// training progress visualization module
class TrainingVisualization {
    constructor() {
        this.model = null;
    }
    
    render(model) {
        console.log('rendering training visualization...');
        
        this.model = model;
        
        if (model.training_history && Array.isArray(model.training_history.train_loss) && model.training_history.train_loss.length) {
            this.plotTrainingCurves();
            this.displayTrainingStats();
        } else {
            this.showNoDataMessage();
        }
        
        console.log('training visualization ready');
    }
    
    plotTrainingCurves() {
        const history = this.model.training_history;
        
        if (!history || !Array.isArray(history.train_loss) || !history.train_loss.length) {
            this.showNoDataMessage();
            return;
        }
        document.getElementById('training-plots').innerHTML = '<div id="loss-plot"></div>';
        
        // create loss curve
        const lossTrace = {
            x: Array.from({length: history.train_loss.length}, (_, i) => i + 1),
            y: history.train_loss,
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
        if (Array.isArray(history.val_loss) && history.val_loss.length) {
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
            ...(history.learning_rate ? {
                yaxis2: {
                    title: 'learning rate',
                    overlaying: 'y',
                    side: 'right',
                    type: 'log'
                }
            } : {}),
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
        
        Plotly.newPlot('loss-plot', traces, layout, config);
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
        const windowSize = Math.max(1, Math.min(10, Math.floor(history.train_loss.length / 10)));
        const convergenceRate = this.computeConvergenceRate(history.train_loss, windowSize);
        
        const convergenceTrace = {
            x: Array.from({length: convergenceRate.length}, (_, i) => i + windowSize + 1),
            y: convergenceRate,
            type: 'scatter',
            mode: 'lines+markers',
            name: 'relative rolling-mean loss decrease',
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
            title: 'signed relative decrease in rolling-mean training loss',
            xaxis: {
                title: 'epoch',
                gridcolor: '#eee'
            },
            yaxis: {
                title: 'relative decrease',
                gridcolor: '#eee'
            },
            ...(history.grad_norm ? {
                yaxis2: {
                    title: 'gradient norm',
                    overlaying: 'y',
                    side: 'right',
                    type: 'log'
                }
            } : {}),
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
        
        for (let i = windowSize + 1; i <= losses.length; i++) {
            const currentWindow = losses.slice(i - windowSize, i);
            const previousWindow = losses.slice(i - windowSize - 1, i - 1);
            const currentAvg = currentWindow.reduce((a, b) => a + b, 0) / windowSize;
            const previousAvg = previousWindow.reduce((a, b) => a + b, 0) / windowSize;
            rates.push(previousAvg === 0 ? null : (previousAvg - currentAvg) / previousAvg);
        }
        
        return rates;
    }
    
    displayTrainingStats() {
        const history = this.model.training_history;
        const metadata = this.model.metadata;
        const statsContainer = document.getElementById('stats-content');
        
        // compute training statistics
        const finalLoss = history.train_loss[history.train_loss.length - 1];
        const initialLoss = history.train_loss[0];
        const improvementRatio = initialLoss / finalLoss;
        const totalEpochs = history.train_loss.length;
        
        // find best epoch
        const bestEpoch = history.train_loss.indexOf(Math.min(...history.train_loss)) + 1;
        const bestLoss = Math.min(...history.train_loss);
        
        // compute convergence info
        const convergenceThreshold = initialLoss * 0.01; // 1% of initial loss
        const convergedEpoch = history.train_loss.findIndex(loss => loss <= convergenceThreshold);
        
        // training speed
        const avgLossReduction = (initialLoss - finalLoss) / Math.max(1, totalEpochs - 1);
        
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
                <span>initial / final loss:</span>
                <strong>${improvementRatio.toFixed(1)}×</strong>
            </div>
            <div class="stat-item">
                <span>first loss ≤ 1% of initial:</span>
                <strong>${convergedEpoch >= 0 ? `epoch ${convergedEpoch + 1}` : 'not reached'}</strong>
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
                <strong>model parameters:</strong><br>
                • ${metadata.total_parameters} total parameters<br>
                • ${metadata.num_layers} layers<br>
                • Gaussian RBF basis<br>
                • ${metadata.grid_size} grid intervals / ${metadata.grid_size + 1} centers
            </div>
        `;
        
        statsContainer.innerHTML += `
            <p>these are recorded training losses. a training-loss decrease alone
            does not establish generalization or show whether the model needs more capacity.</p>
        `;
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