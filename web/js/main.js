// main application controller
class KANVisualizer {
    constructor() {
        this.currentModel = null;
        this.currentModelName = 'model_1d';
        this.currentMode = 'network';
        this.datasets = null;
        this.motionPreference = window.matchMedia('(prefers-reduced-motion: reduce)');
        
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
        document.querySelectorAll('.guide-steps button').forEach(button => {
            button.addEventListener('click', () => {
                this.currentMode = button.dataset.view;
                document.getElementById('visualization-mode').value = this.currentMode;
                this.switchMode();
            });
        });
        this.motionPreference.addEventListener('change', () => {
            this.networkViz.stopAnimation();
            this.inferenceEngine.stopAnimation();
            this.isAnimating = false;
            this.updateAnimationControl();
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
        this.updateAnimationControl();
        this.updateGuide();
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

    updateGuide() {
        const explanations = {
            network: 'I read left to right: inputs, hidden nodes, then output. I select a node or edge in the graph or the labelled menus. Width summarizes sampled edge RMS, not the contribution at a particular input.',
            splines: 'I choose a layer and connection to inspect its full sampled edge function: base activation plus Gaussian RBF branch. The diamonds mark Gaussian centers at zero; they are not measured contributions.',
            inference: 'I move a slider with the arrow keys to recompute the exported model. Labels show actual activations, and edges show signed contributions at this input. I compare the prediction with the known synthetic target.',
            training: 'I read the loss recorded during training, not a live optimization. The second chart shows the signed relative decrease in rolling-mean loss; a negative value means the loss rose.'
        };
        document.getElementById('view-explanation').textContent = explanations[this.currentMode];
        document.querySelectorAll('.guide-steps button').forEach(button => {
            if (button.dataset.view === this.currentMode) button.setAttribute('aria-current', 'step');
            else button.removeAttribute('aria-current');
        });
    }

    updateAnimationControl() {
        const button = document.getElementById('play-button');
        button.disabled = this.motionPreference.matches || !['network', 'inference'].includes(this.currentMode);
        button.setAttribute('aria-pressed', String(Boolean(this.isAnimating)));
        button.textContent = this.motionPreference.matches
            ? 'Reduced motion: use manual controls'
            : this.isAnimating ? 'Pause animation' : this.animationLabel();
    }

    animationLabel() {
        return this.currentMode === 'inference' ? 'Sweep inputs (real inference)' : 'Illustrate connectivity';
    }

    playAnimation() {
        if (this.motionPreference.matches || !['network', 'inference'].includes(this.currentMode)) return;
        const visualization = this.currentMode === 'network' ? this.networkViz : this.inferenceEngine;
        this.isAnimating = !this.isAnimating;
        this.updateAnimationControl();
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