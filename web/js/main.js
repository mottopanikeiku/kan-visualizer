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
        
        // load initial data
        await this.loadData();
        
        // hide loading overlay
        this.hideLoading();
        
        console.log('kan visualizer ready!');
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
            await this.loadModel();
            
        } catch (error) {
            console.error('error loading data:', error);
            this.showError('failed to load data. please check that the json files exist.');
        }
    }
    
    async loadModel() {
        try {
            console.log(`loading model: ${this.currentModelName}`);
            
            const response = await fetch(`data/${this.currentModelName}.json`);
            this.currentModel = await response.json();
            
            console.log('model loaded:', this.currentModel);
            
            // update ui
            this.updateModelInfo();
            
            // refresh current visualization
            this.switchMode();
            
        } catch (error) {
            console.error('error loading model:', error);
            this.showError(`failed to load model: ${this.currentModelName}`);
        }
    }
    
    updateModelInfo() {
        const info = this.currentModel.metadata;
        const infoText = `${info.num_layers} layers | ${info.total_parameters} parameters | grid size: ${info.grid_size}`;
        document.getElementById('network-info').textContent = infoText;
    }
    
    switchMode() {
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
        // return appropriate dataset based on model input dimensions
        const inputDim = this.currentModel.metadata.architecture[0];
        
        if (inputDim === 1) {
            return this.datasets['1d_sine_wave'];
        } else if (inputDim === 2) {
            return this.datasets['2d_gaussian'];
        }
        
        return null;
    }
    
    playAnimation() {
        const button = document.getElementById('play-button');
        
        if (button.textContent.includes('▶')) {
            // start animation
            button.textContent = '⏸ pause animation';
            
            switch (this.currentMode) {
                case 'network':
                    this.networkViz.startAnimation();
                    break;
                case 'inference':
                    this.inferenceEngine.startAnimation();
                    break;
            }
        } else {
            // stop animation
            button.textContent = '▶ animate data flow';
            
            switch (this.currentMode) {
                case 'network':
                    this.networkViz.stopAnimation();
                    break;
                case 'inference':
                    this.inferenceEngine.stopAnimation();
                    break;
            }
        }
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