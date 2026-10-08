// helpers shared by the network and edge-function views
class Utils {
    // Exported samples contain the full edge contribution, including both scales.
    static edgeSamples(layer, inputIdx, outputIdx) {
        const data = layer.edge_evaluations && layer.edge_evaluations.find(
            entry => entry.input_idx === inputIdx && entry.output_idx === outputIdx
        );
        if (!data || !Array.isArray(data.x_values) || !Array.isArray(data.y_values) ||
            data.x_values.length < 2 || data.x_values.length !== data.y_values.length ||
            !data.x_values.every(Number.isFinite) || !data.y_values.every(Number.isFinite)) {
            throw new Error(`Missing or invalid full edge samples: input ${inputIdx}, output ${outputIdx}`);
        }
        return data;
    }

    static sampleRms(values) {
        return Math.sqrt(values.reduce((sum, value) => sum + value * value, 0) / values.length);
    }
}

// global error handler; event.error is null for cross-origin script and resource errors
window.addEventListener('error', (event) => {
    console.error('kan visualizer error:', event.error || event.message);
    
    // show user-friendly error message
    const errorDiv = document.createElement('div');
    errorDiv.setAttribute('role', 'alert');
    errorDiv.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        background: #ff6b6b;
        color: white;
        padding: 15px;
        border-radius: 8px;
        z-index: 10000;
        max-width: 300px;
        font-size: 14px;
    `;
    const title = document.createElement('strong');
    title.textContent = 'error occurred';
    const message = document.createElement('p');
    message.textContent = (event.error && event.error.message) || event.message || 'something went wrong';
    const hint = document.createElement('small');
    hint.textContent = 'check console for details';
    errorDiv.append(title, message, hint);
    
    document.body.appendChild(errorDiv);
    
    // auto-remove after 5 seconds
    setTimeout(() => {
        if (errorDiv.parentNode) {
            errorDiv.parentNode.removeChild(errorDiv);
        }
    }, 5000);
});