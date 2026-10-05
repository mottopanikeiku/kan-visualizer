// main application controller
class KANVisualizer {
    constructor() {
        this.currentModel = null;
        this.currentModelName = 'model_1d';
        this.currentMode = 'network';
        this.datasets = null;
        
        this.networkViz = new NetworkVisualization();
        this.splineViz = new SplineVisualization();
        this.inferenceEngine = new InferenceEngine();
        this.trainingViz = new TrainingVisualization();
        
        this.init();
    }
    
    async init() {
        console.log('initializing kan visualizer...');
        
        // setup event listeners
        this.setupEventListeners();
        
        if (await this.loadData()) {
            console.log('kan visualizer ready!');
        }
    }
    
    setupEventListeners() {
        // model selection
        document.getElementById('model-select').addEventListener('change', (e) => {
            this.currentModelName = e.target.value;
            this.loadModel();
        });
        
        // visualization mode
        document.getElementById('visualization-mode').addEventListener('change', (e) => {
            this.currentMode = e.target.value;
            this.switchMode();
        });
        
        // play button
        document.getElementById('play-button').addEventListener('click', () => {
            this.playAnimation();
        });
    }
    
    async loadData() {
        try {
            // load datasets
            const datasetsResponse = await fetch('data/datasets.json');
            this.datasets = await datasetsResponse.json();
            
            // load initial model
            return await this.loadModel();
            
        } catch (error) {
            console.error('error loading data:', error);
            this.showError('failed to load data. please check that the json files exist.');
            return false;
        }
    }
    
    async loadModel() {
        try {
            console.log(`loading model: ${this.currentModelName}`);
            this.networkViz.stopAnimation();
            this.inferenceEngine.stopAnimation();
            
            const response = await fetch(`data/${this.currentModelName}.json`);
            this.currentModel = await response.json();
            KANForward.forward(this.currentModel, new Array(this.currentModel.metadata.architecture[0]).fill(0));
            this.getDatasetForModel();
            
            console.log('model loaded:', this.currentModel);
            
            // update ui
            this.updateModelInfo();
            
            // refresh current visualization
            this.switchMode();
            this.hideLoading();
            return true;
            
        } catch (error) {
            console.error('error loading model:', error);
            this.showError(`failed to load model: ${this.currentModelName}. ${error.message}`);
            return false;
        }
    }
    
    updateModelInfo() {
        const info = this.currentModel.metadata;
        const infoText = `${info.num_layers} layers | ${info.total_parameters} parameters | Gaussian RBF | ${info.grid_size} intervals / ${info.grid_size + 1} centers`;
        document.getElementById('network-info').textContent = infoText;
    }
    
    switchMode() {
        this.networkViz.stopAnimation();
        this.inferenceEngine.stopAnimation();
        this.isAnimating = false;
        const button = document.getElementById('play-button');
        button.disabled = !['network', 'inference'].includes(this.currentMode);
        button.textContent = this.animationLabel();
        // hide all panels
        document.querySelectorAll('.panel').forEach(panel => {
            panel.classList.remove('active');
        });
        
        // show current panel
        const panel = document.getElementById(`${this.currentMode}-panel`);
        if (panel) {
            panel.classList.add('active');
        }
        
        // initialize appropriate visualization
        switch (this.currentMode) {
            case 'network':
                this.networkViz.render(this.currentModel);
                break;
            case 'splines':
                this.splineViz.render(this.currentModel);
                break;
            case 'inference':
                this.inferenceEngine.render(this.currentModel, this.getDatasetForModel());
                break;
            case 'training':
                this.trainingViz.render(this.currentModel);
                break;
        }
    }
    
    getDatasetForModel() {
        const targetId = this.currentModel.metadata.target_id;
        const dataset = this.datasets && this.datasets[targetId];
        if (!dataset || dataset.target_id !== targetId) {
            throw new Error(`No matching dataset for target_id: ${targetId}`);
        }
        return dataset;
    }

    animationLabel() {
        return this.currentMode === 'inference' ? '▶ sweep inputs (real inference)' : '▶ illustrate connectivity';
    }

    playAnimation() {
        if (!['network', 'inference'].includes(this.currentMode)) return;
        const visualization = this.currentMode === 'network' ? this.networkViz : this.inferenceEngine;
        this.isAnimating = !this.isAnimating;
        document.getElementById('play-button').textContent =
            this.isAnimating ? '⏸ pause animation' : this.animationLabel();
        if (this.isAnimating) visualization.startAnimation();
        else visualization.stopAnimation();
    }
    
    hideLoading() {
        const overlay = document.getElementById('loading-overlay');
        overlay.style.opacity = '0';
        setTimeout(() => {
            overlay.style.display = 'none';
        }, 300);
    }
    
    showError(message) {
        const overlay = document.getElementById('loading-overlay');
        overlay.style.display = 'flex';
        overlay.style.opacity = '1';
        overlay.innerHTML = `
            <div style="text-align: center;">
                <h2>⚠️ error</h2>
                <p>${message}</p>
                <button onclick="location.reload()" style="margin-top: 20px; padding: 10px 20px; background: white; color: #667eea; border: none; border-radius: 5px; cursor: pointer;">
                    reload page
                </button>
            </div>
        `;
    }
}

// initialize app when page loads
document.addEventListener('DOMContentLoaded', () => {
    window.kanApp = new KANVisualizer();
});