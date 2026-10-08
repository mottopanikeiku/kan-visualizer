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
    
    // generate evenly spaced function samples
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