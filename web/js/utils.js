// utility functions for kan visualization
class Utils {
    // format numbers for display
    static formatNumber(num, decimals = 3) {
        if (Math.abs(num) < 1e-10) return '0';
        if (Math.abs(num) < 1e-3 || Math.abs(num) > 1e3) {
            return parseFloat(num).toExponential(decimals);
        }
        return parseFloat(num).toFixed(decimals);
    }
    
    // interpolate between colors
    static interpolateColor(color1, color2, factor) {
        const c1 = d3.color(color1);
        const c2 = d3.color(color2);
        return d3.interpolate(c1, c2)(factor);
    }
    
    // evaluate simplified spline function
    static evaluateSpline(gridPoints, coefficients, x) {
        // clamp x to grid range
        x = Math.max(gridPoints[0], Math.min(gridPoints[gridPoints.length - 1], x));
        
        // find interval
        let i = 0;
        while (i < gridPoints.length - 1 && x > gridPoints[i + 1]) {
            i++;
        }
        
        // linear interpolation for simplicity
        if (i >= coefficients.length - 1) i = coefficients.length - 2;
        const t = (x - gridPoints[i]) / (gridPoints[i + 1] - gridPoints[i]);
        return coefficients[i] * (1 - t) + coefficients[i + 1] * t;
    }
    
    // create svg gradients
    static createGradient(svg, id, colors) {
        const gradient = svg.append('defs')
            .append('linearGradient')
            .attr('id', id)
            .attr('gradientUnits', 'userSpaceOnUse');
        
        colors.forEach((color, i) => {
            gradient.append('stop')
                .attr('offset', `${(i / (colors.length - 1)) * 100}%`)
                .attr('stop-color', color);
        });
        
        return gradient;
    }
    
    // normalize array to [0, 1]
    static normalize(array) {
        const min = Math.min(...array);
        const max = Math.max(...array);
        const range = max - min;
        
        if (range === 0) return array.map(() => 0);
        return array.map(x => (x - min) / range);
    }
    
    // compute moving average
    static movingAverage(array, windowSize) {
        const result = [];
        for (let i = 0; i < array.length; i++) {
            const start = Math.max(0, i - Math.floor(windowSize / 2));
            const end = Math.min(array.length, start + windowSize);
            const window = array.slice(start, end);
            const avg = window.reduce((a, b) => a + b) / window.length;
            result.push(avg);
        }
        return result;
    }
    
    // generate color scale
    static generateColorScale(domain, range) {
        return d3.scaleLinear()
            .domain(domain)
            .range(range);
    }
    
    // animate number counting
    static animateNumber(element, start, end, duration = 1000) {
        const range = end - start;
        const increment = range / (duration / 16); // ~60fps
        let current = start;
        
        const timer = setInterval(() => {
            current += increment;
            if ((increment > 0 && current >= end) || (increment < 0 && current <= end)) {
                current = end;
                clearInterval(timer);
            }
            element.textContent = Utils.formatNumber(current);
        }, 16);
    }
    
    // debounce function calls
    static debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    }
    
    // show tooltip
    static showTooltip(x, y, content) {
        let tooltip = document.getElementById('kan-tooltip');
        if (!tooltip) {
            tooltip = document.createElement('div');
            tooltip.id = 'kan-tooltip';
            tooltip.style.cssText = `
                position: absolute;
                background: rgba(0,0,0,0.8);
                color: white;
                padding: 8px 12px;
                border-radius: 4px;
                font-size: 12px;
                pointer-events: none;
                z-index: 10000;
                opacity: 0;
                transition: opacity 0.2s;
            `;
            document.body.appendChild(tooltip);
        }
        
        tooltip.innerHTML = content;
        tooltip.style.left = x + 10 + 'px';
        tooltip.style.top = y - 10 + 'px';
        tooltip.style.opacity = '1';
    }
    
    // hide tooltip
    static hideTooltip() {
        const tooltip = document.getElementById('kan-tooltip');
        if (tooltip) {
            tooltip.style.opacity = '0';
        }
    }
    
    // download data as json
    static downloadJSON(data, filename) {
        const blob = new Blob([JSON.stringify(data, null, 2)], {
            type: 'application/json'
        });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }
    
    // copy text to clipboard
    static copyToClipboard(text) {
        if (navigator.clipboard) {
            navigator.clipboard.writeText(text);
        } else {
            // fallback for older browsers
            const textarea = document.createElement('textarea');
            textarea.value = text;
            document.body.appendChild(textarea);
            textarea.select();
            document.execCommand('copy');
            document.body.removeChild(textarea);
        }
    }
    
    // check if device is mobile
    static isMobile() {
        return window.innerWidth <= 768;
    }
    
    // smooth scrolling to element
    static scrollToElement(element, offset = 0) {
        const elementPosition = element.getBoundingClientRect().top + window.pageYOffset;
        const offsetPosition = elementPosition - offset;
        
        window.scrollTo({
            top: offsetPosition,
            behavior: 'smooth'
        });
    }
    
    // generate random data for testing
    static generateTestData(func, xRange, nPoints) {
        const data = { x: [], y: [] };
        for (let i = 0; i < nPoints; i++) {
            const x = xRange[0] + (xRange[1] - xRange[0]) * i / (nPoints - 1);
            const y = func(x);
            data.x.push(x);
            data.y.push(y);
        }
        return data;
    }
    
    // compute statistics
    static computeStats(array) {
        const sorted = [...array].sort((a, b) => a - b);
        const mean = array.reduce((a, b) => a + b) / array.length;
        const variance = array.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / array.length;
        
        return {
            min: Math.min(...array),
            max: Math.max(...array),
            mean: mean,
            median: sorted[Math.floor(sorted.length / 2)],
            std: Math.sqrt(variance),
            q25: sorted[Math.floor(sorted.length * 0.25)],
            q75: sorted[Math.floor(sorted.length * 0.75)]
        };
    }
}

// global error handler
window.addEventListener('error', (event) => {
    console.error('kan visualizer error:', event.error);
    
    // show user-friendly error message
    const errorDiv = document.createElement('div');
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
    errorDiv.innerHTML = `
        <strong>⚠️ error occurred</strong><br>
        ${event.error.message || 'something went wrong'}<br>
        <small>check console for details</small>
    `;
    
    document.body.appendChild(errorDiv);
    
    // auto-remove after 5 seconds
    setTimeout(() => {
        if (errorDiv.parentNode) {
            errorDiv.parentNode.removeChild(errorDiv);
        }
    }, 5000);
});

// keyboard shortcuts
document.addEventListener('keydown', (event) => {
    // ctrl/cmd + h: toggle help
    if ((event.ctrlKey || event.metaKey) && event.key === 'h') {
        event.preventDefault();
        Utils.showHelp();
    }
    
    // ctrl/cmd + d: download current model data
    if ((event.ctrlKey || event.metaKey) && event.key === 'd') {
        event.preventDefault();
        if (window.kanViz && window.kanViz.currentModel) {
            Utils.downloadJSON(window.kanViz.currentModel, 'kan_model.json');
        }
    }
});

// add help functionality
Utils.showHelp = function() {
    const helpDiv = document.createElement('div');
    helpDiv.style.cssText = `
        position: fixed;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        background: white;
        padding: 30px;
        border-radius: 15px;
        box-shadow: 0 20px 40px rgba(0,0,0,0.3);
        z-index: 10000;
        max-width: 500px;
        max-height: 80vh;
        overflow-y: auto;
    `;
    
    helpDiv.innerHTML = `
        <h2 style="margin-top: 0; color: #667eea;">🧠 kan visualizer help</h2>
        
        <h3>navigation</h3>
        <ul>
            <li><strong>network view:</strong> explore the overall architecture</li>
            <li><strong>splines view:</strong> examine individual learned functions</li>
            <li><strong>inference view:</strong> test the model with live inputs</li>
            <li><strong>training view:</strong> analyze learning progress</li>
        </ul>
        
        <h3>interactions</h3>
        <ul>
            <li><strong>click nodes/edges:</strong> show detailed information</li>
            <li><strong>hover elements:</strong> highlight connections</li>
            <li><strong>drag sliders:</strong> change input values in real-time</li>
            <li><strong>play button:</strong> animate data flow</li>
        </ul>
        
        <h3>keyboard shortcuts</h3>
        <ul>
            <li><strong>ctrl/cmd + h:</strong> show this help</li>
            <li><strong>ctrl/cmd + d:</strong> download model data</li>
        </ul>
        
        <h3>understanding kans</h3>
        <p>kolmogorov-arnold networks replace linear weights with learnable spline functions. 
        each edge represents a univariate function that the network learns to approximate 
        complex multivariate functions.</p>
        
        <button onclick="this.parentElement.remove()" 
                style="margin-top: 20px; padding: 10px 20px; background: #667eea; 
                       color: white; border: none; border-radius: 5px; cursor: pointer;">
            close
        </button>
    `;
    
    // add backdrop
    const backdrop = document.createElement('div');
    backdrop.style.cssText = `
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(0,0,0,0.5);
        z-index: 9999;
    `;
    backdrop.onclick = () => {
        backdrop.remove();
        helpDiv.remove();
    };
    
    document.body.appendChild(backdrop);
    document.body.appendChild(helpDiv);
}; 