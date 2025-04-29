/**
 * Analysis Dashboard JavaScript
 * 
 * This file contains the JavaScript code for the analysis dashboard.
 */

// Global variables
const socket = io();
const charts = {};
const chartData = {};
const chartConfigs = {};
const activeSubscriptions = new Set();

// Initialize the dashboard when the DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    // Initialize charts
    initializeCharts();
    
    // Load cameras
    loadCameras();
    
    // Load active algorithms
    loadActiveAlgorithms();
    
    // Set up event listeners
    setupEventListeners();
    
    // Connect to Socket.IO
    connectToSocket();
});

/**
 * Initialize all charts on the dashboard
 */
function initializeCharts() {
    // System performance chart
    initializeChart('system-performance-chart', 'line', 'System Performance', ['CPU', 'Memory', 'GPU']);
    
    // Component charts
    initializeChart('detection-chart', 'line', 'Object Detection Performance', []);
    initializeChart('tracking-chart', 'line', 'Object Tracking Performance', []);
    initializeChart('loitering-chart', 'line', 'Loitering Detection Performance', []);
    initializeChart('crowd-chart', 'line', 'Crowd Detection Performance', []);
    initializeChart('preprocessing-chart', 'line', 'Video Preprocessing Performance', []);
    initializeChart('streaming-chart', 'line', 'Streaming Performance', []);
}

/**
 * Initialize a chart with the given ID and type
 * 
 * @param {string} chartId - The ID of the canvas element
 * @param {string} type - The type of chart (line, bar, etc.)
 * @param {string} title - The title of the chart
 * @param {Array} datasets - Initial datasets for the chart
 */
function initializeChart(chartId, type, title, datasets) {
    const ctx = document.getElementById(chartId).getContext('2d');
    
    // Create datasets
    const chartDatasets = datasets.map((label, index) => {
        return {
            label: label,
            data: [],
            borderColor: getColor(index),
            backgroundColor: getColor(index, 0.2),
            borderWidth: 2,
            tension: 0.4,
            pointRadius: 3
        };
    });
    
    // Create chart
    charts[chartId] = new Chart(ctx, {
        type: type,
        data: {
            labels: [],
            datasets: chartDatasets
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: title,
                    font: {
                        size: 16
                    }
                },
                legend: {
                    position: 'top'
                },
                tooltip: {
                    mode: 'index',
                    intersect: false
                }
            },
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Time'
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Value'
                    },
                    beginAtZero: true
                }
            }
        }
    });
    
    // Store chart data
    chartData[chartId] = {
        labels: [],
        datasets: datasets.map(label => [])
    };
    
    // Store chart config
    chartConfigs[chartId] = {
        type: type,
        title: title,
        yAxisLabel: 'Value'
    };
}

/**
 * Load cameras from the API
 */
function loadCameras() {
    // Fetch cameras from the API
    fetch('/api/metrics')
        .then(response => response.json())
        .then(data => {
            // Extract camera names from metrics
            const cameras = new Set();
            
            // Add cameras to select elements
            const cameraSelects = document.querySelectorAll('select[id$="-camera"]');
            cameraSelects.forEach(select => {
                // Clear existing options
                select.innerHTML = '<option value="all">All Cameras</option>';
                
                // Add camera options
                cameras.forEach(camera => {
                    const option = document.createElement('option');
                    option.value = camera;
                    option.textContent = camera;
                    select.appendChild(option);
                });
            });
        })
        .catch(error => console.error('Error loading cameras:', error));
}

/**
 * Load active algorithms from the API
 */
function loadActiveAlgorithms() {
    // Fetch active algorithms from the API
    fetch('/api/algorithms')
        .then(response => response.json())
        .then(data => {
            // Populate active algorithms table
            const tbody = document.getElementById('active-algorithms');
            tbody.innerHTML = '';
            
            // Add rows for each component
            const components = {
                'detection': 'Object Detection',
                'tracking': 'Object Tracking',
                'preprocessing': 'Video Preprocessing',
                'loitering': 'Loitering Detection',
                'crowd': 'Crowd Detection'
            };
            
            for (const [component, label] of Object.entries(components)) {
                const algorithms = data[component] || [];
                if (algorithms.length > 0) {
                    const row = document.createElement('tr');
                    row.innerHTML = `
                        <td>${label}</td>
                        <td>${algorithms[0]}</td>
                        <td><span class="badge bg-success">Active</span></td>
                    `;
                    tbody.appendChild(row);
                }
            }
        })
        .catch(error => console.error('Error loading active algorithms:', error));
}

/**
 * Set up event listeners for the dashboard
 */
function setupEventListeners() {
    // Chart type selectors
    document.querySelectorAll('select[id$="-chart-type"]').forEach(select => {
        select.addEventListener('change', function() {
            const component = this.id.split('-')[0];
            const chartId = `${component}-chart`;
            updateChartType(chartId, this.value);
        });
    });
    
    // Metric selectors
    document.querySelectorAll('select[id$="-metric"]').forEach(select => {
        select.addEventListener('change', function() {
            const component = this.id.split('-')[0];
            const chartId = `${component}-chart`;
            const metric = this.value;
            
            // Update chart title and y-axis label
            updateChartConfig(chartId, {
                title: `${component.charAt(0).toUpperCase() + component.slice(1)} ${metric.charAt(0).toUpperCase() + metric.slice(1)}`,
                yAxisLabel: metric.charAt(0).toUpperCase() + metric.slice(1)
            });
            
            // Subscribe to new metric
            subscribeToMetric(component, metric);
        });
    });
    
    // Algorithm selectors
    document.querySelectorAll('select[id$="-algorithm"]').forEach(select => {
        select.addEventListener('change', function() {
            const component = this.id.split('-')[0];
            const algorithm = this.value;
            
            // Update comparison table
            updateComparisonTable(component, algorithm);
        });
    });
    
    // Camera selectors
    document.querySelectorAll('select[id$="-camera"]').forEach(select => {
        select.addEventListener('change', function() {
            const component = this.id.split('-')[0];
            const metric = document.getElementById(`${component}-metric`).value;
            const camera = this.value;
            
            // Subscribe to new metric
            subscribeToMetric(component, metric, camera);
        });
    });
}

/**
 * Connect to Socket.IO and set up event handlers
 */
function connectToSocket() {
    // Connection event
    socket.on('connect', () => {
        console.log('Connected to Socket.IO server');
        
        // Subscribe to initial metrics
        subscribeToInitialMetrics();
    });
    
    // Disconnection event
    socket.on('disconnect', () => {
        console.log('Disconnected from Socket.IO server');
    });
    
    // Metric update event
    socket.on('metric_update', (data) => {
        updateMetricData(data);
    });
}

/**
 * Subscribe to initial metrics
 */
function subscribeToInitialMetrics() {
    // System performance metrics
    subscribeToMetric('system', 'cpu_usage');
    subscribeToMetric('system', 'memory_usage');
    subscribeToMetric('system', 'gpu_usage');
    
    // Component metrics
    const components = ['detection', 'tracking', 'loitering', 'crowd', 'preprocessing', 'streaming'];
    components.forEach(component => {
        const metricSelect = document.getElementById(`${component}-metric`);
        if (metricSelect) {
            subscribeToMetric(component, metricSelect.value);
        }
    });
}

/**
 * Subscribe to a metric
 * 
 * @param {string} component - The component name
 * @param {string} metric - The metric name
 * @param {string} camera - The camera name (optional)
 */
function subscribeToMetric(component, metric, camera = 'all') {
    // Create subscription key
    const subscriptionKey = `${component}_${metric}_${camera}`;
    
    // Unsubscribe from existing subscriptions for this component
    unsubscribeFromComponent(component);
    
    // Add new subscription
    activeSubscriptions.add(subscriptionKey);
    
    // Send subscription to server
    socket.emit('subscribe', {
        component: component,
        metric: metric,
        camera: camera === 'all' ? null : camera,
        interval: 1
    });
    
    // Clear existing chart data
    const chartId = `${component}-chart`;
    if (charts[chartId]) {
        chartData[chartId].labels = [];
        chartData[chartId].datasets = [[]];
        updateChart(chartId);
    }
}

/**
 * Unsubscribe from all metrics for a component
 * 
 * @param {string} component - The component name
 */
function unsubscribeFromComponent(component) {
    // Find all subscriptions for this component
    const subscriptionsToRemove = [];
    activeSubscriptions.forEach(key => {
        if (key.startsWith(`${component}_`)) {
            subscriptionsToRemove.push(key);
            
            // Parse subscription key
            const [comp, metric, camera] = key.split('_');
            
            // Send unsubscribe to server
            socket.emit('unsubscribe', {
                component: comp,
                metric: metric,
                camera: camera === 'all' ? null : camera,
                interval: 1
            });
        }
    });
    
    // Remove subscriptions
    subscriptionsToRemove.forEach(key => activeSubscriptions.delete(key));
}

/**
 * Update metric data when a new value is received
 * 
 * @param {Object} data - The metric data
 */
function updateMetricData(data) {
    const { component, metric, camera, value, timestamp } = data;
    
    // Format timestamp
    const date = new Date(timestamp * 1000);
    const timeString = date.toLocaleTimeString();
    
    // Update chart data
    const chartId = `${component}-chart`;
    if (charts[chartId]) {
        // Add label
        chartData[chartId].labels.push(timeString);
        if (chartData[chartId].labels.length > 20) {
            chartData[chartId].labels.shift();
        }
        
        // Add data point
        chartData[chartId].datasets[0].push(value);
        if (chartData[chartId].datasets[0].length > 20) {
            chartData[chartId].datasets[0].shift();
        }
        
        // Update chart
        updateChart(chartId);
    }
    
    // Update system performance chart
    if (component === 'system') {
        const index = metric === 'cpu_usage' ? 0 : (metric === 'memory_usage' ? 1 : 2);
        
        // Add label if needed
        if (index === 0) {
            chartData['system-performance-chart'].labels.push(timeString);
            if (chartData['system-performance-chart'].labels.length > 20) {
                chartData['system-performance-chart'].labels.shift();
            }
        }
        
        // Add data point
        chartData['system-performance-chart'].datasets[index].push(value);
        if (chartData['system-performance-chart'].datasets[index].length > 20) {
            chartData['system-performance-chart'].datasets[index].shift();
        }
        
        // Update chart
        updateChart('system-performance-chart');
    }
    
    // Update comparison tables
    updateComparisonTableValue(component, metric, value);
}

/**
 * Update a chart with the latest data
 * 
 * @param {string} chartId - The ID of the chart to update
 */
function updateChart(chartId) {
    if (charts[chartId]) {
        charts[chartId].data.labels = chartData[chartId].labels;
        
        // Update datasets
        for (let i = 0; i < chartData[chartId].datasets.length; i++) {
            if (charts[chartId].data.datasets[i]) {
                charts[chartId].data.datasets[i].data = chartData[chartId].datasets[i];
            }
        }
        
        // Update chart
        charts[chartId].update();
    }
}

/**
 * Update the chart type
 * 
 * @param {string} chartId - The ID of the chart to update
 * @param {string} type - The new chart type
 */
function updateChartType(chartId, type) {
    if (charts[chartId]) {
        // Store current data
        const data = charts[chartId].data;
        
        // Destroy current chart
        charts[chartId].destroy();
        
        // Create new chart with new type
        const ctx = document.getElementById(chartId).getContext('2d');
        charts[chartId] = new Chart(ctx, {
            type: type,
            data: data,
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: chartConfigs[chartId].title,
                        font: {
                            size: 16
                        }
                    },
                    legend: {
                        position: 'top'
                    },
                    tooltip: {
                        mode: 'index',
                        intersect: false
                    }
                },
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: 'Time'
                        }
                    },
                    y: {
                        title: {
                            display: true,
                            text: chartConfigs[chartId].yAxisLabel
                        },
                        beginAtZero: true
                    }
                }
            }
        });
        
        // Update chart config
        chartConfigs[chartId].type = type;
    }
}

/**
 * Update the chart configuration
 * 
 * @param {string} chartId - The ID of the chart to update
 * @param {Object} config - The new configuration
 */
function updateChartConfig(chartId, config) {
    if (charts[chartId]) {
        // Update title
        if (config.title) {
            charts[chartId].options.plugins.title.text = config.title;
            chartConfigs[chartId].title = config.title;
        }
        
        // Update y-axis label
        if (config.yAxisLabel) {
            charts[chartId].options.scales.y.title.text = config.yAxisLabel;
            chartConfigs[chartId].yAxisLabel = config.yAxisLabel;
        }
        
        // Update chart
        charts[chartId].update();
    }
}

/**
 * Update the comparison table for a component
 * 
 * @param {string} component - The component name
 * @param {string} algorithm - The algorithm name
 */
function updateComparisonTable(component, algorithm) {
    // TODO: Implement comparison table update
    console.log(`Updating comparison table for ${component} with algorithm ${algorithm}`);
}

/**
 * Update a value in the comparison table
 * 
 * @param {string} component - The component name
 * @param {string} metric - The metric name
 * @param {number} value - The metric value
 */
function updateComparisonTableValue(component, metric, value) {
    // TODO: Implement comparison table value update
    console.log(`Updating comparison table value for ${component} ${metric}: ${value}`);
}

/**
 * Get a color for a dataset
 * 
 * @param {number} index - The index of the dataset
 * @param {number} alpha - The alpha value for the color
 * @returns {string} - The color string
 */
function getColor(index, alpha = 1) {
    const colors = [
        `rgba(54, 162, 235, ${alpha})`,   // Blue
        `rgba(255, 99, 132, ${alpha})`,   // Red
        `rgba(75, 192, 192, ${alpha})`,   // Green
        `rgba(255, 159, 64, ${alpha})`,   // Orange
        `rgba(153, 102, 255, ${alpha})`,  // Purple
        `rgba(255, 205, 86, ${alpha})`,   // Yellow
        `rgba(201, 203, 207, ${alpha})`   // Grey
    ];
    
    return colors[index % colors.length];
}
