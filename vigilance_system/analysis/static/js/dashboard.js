// Analysis Dashboard JavaScript

// Global variables
let socket;
let charts = {};
let currentAlgorithms = {};
let availableAlgorithms = {};
let selectedCamera = '';
let metricsData = {};

// Chart colors
const chartColors = {
    blue: 'rgba(54, 162, 235, 0.5)',
    green: 'rgba(75, 192, 192, 0.5)',
    red: 'rgba(255, 99, 132, 0.5)',
    orange: 'rgba(255, 159, 64, 0.5)',
    purple: 'rgba(153, 102, 255, 0.5)',
    yellow: 'rgba(255, 205, 86, 0.5)',
    grey: 'rgba(201, 203, 207, 0.5)'
};

// Component-specific metrics
const componentMetrics = {
    detection: ['fps', 'precision', 'recall', 'map'],
    tracking: ['fps', 'id_switches', 'mota', 'motp'],
    loitering: ['true_positives', 'false_positives', 'false_negatives', 'precision', 'recall'],
    crowd: ['mae', 'mse', 'accuracy', 'precision', 'recall', 'event_count'],
    preprocessing: ['processing_time', 'stability_score']
};

// Initialize the dashboard
document.addEventListener('DOMContentLoaded', function() {
    // Initialize Socket.IO connection
    initializeSocket();

    // Load algorithms and current selections
    loadAlgorithms();

    // Initialize charts
    initializeCharts();

    // Set up event listeners
    setupEventListeners();

    // Start metrics updates
    startMetricsUpdates();
});

// Initialize Socket.IO connection
function initializeSocket() {
    // Get the host from the current URL
    const host = window.location.host;

    // Create Socket.IO connection
    socket = io();

    // Socket.IO event handlers
    socket.on('connect', function() {
        console.log('Connected to server');
    });

    socket.on('disconnect', function() {
        console.log('Disconnected from server');
    });

    socket.on('metrics_update', function(data) {
        updateMetricsData(data);
        updateCharts();
        updateCurrentPerformance();
    });
}

// Load algorithms and current selections
function loadAlgorithms() {
    // Fetch available algorithms
    fetch('/api/algorithms')
        .then(response => response.json())
        .then(data => {
            availableAlgorithms = data;
            populateAlgorithmSelects();
        })
        .catch(error => console.error('Error loading algorithms:', error));

    // Fetch current algorithms
    fetch('/api/current_algorithms')
        .then(response => response.json())
        .then(data => {
            currentAlgorithms = data;
            updateAlgorithmSelects();
        })
        .catch(error => console.error('Error loading current algorithms:', error));
}

// Populate algorithm select elements
function populateAlgorithmSelects() {
    // Detection algorithms
    const detectionSelect = document.getElementById('detectionAlgorithm');
    populateSelect(detectionSelect, availableAlgorithms.detection);

    // Tracking algorithms
    const trackingSelect = document.getElementById('trackingAlgorithm');
    populateSelect(trackingSelect, availableAlgorithms.tracking);

    // Loitering algorithms
    const loiteringSelect = document.getElementById('loiteringAlgorithm');
    populateSelect(loiteringSelect, availableAlgorithms.loitering);

    // Crowd algorithms
    const crowdSelect = document.getElementById('crowdAlgorithm');
    populateSelect(crowdSelect, availableAlgorithms.crowd);

    // Preprocessing algorithms
    const preprocessingSelect = document.getElementById('preprocessingAlgorithm');
    populateSelect(preprocessingSelect, availableAlgorithms.preprocessing);

    // Comparison metrics
    updateComparisonMetrics();
}

// Populate a select element with options
function populateSelect(selectElement, options) {
    // Clear existing options
    selectElement.innerHTML = '';

    // Add options
    if (options) {
        options.forEach(option => {
            const optionElement = document.createElement('option');
            optionElement.value = option;
            optionElement.textContent = formatAlgorithmName(option);
            selectElement.appendChild(optionElement);
        });
    }
}

// Format algorithm name for display
function formatAlgorithmName(name) {
    // Convert snake_case to Title Case
    return name.split('_').map(word =>
        word.charAt(0).toUpperCase() + word.slice(1)
    ).join(' ');
}

// Update algorithm select elements with current selections
function updateAlgorithmSelects() {
    // Detection algorithm
    const detectionSelect = document.getElementById('detectionAlgorithm');
    if (detectionSelect && currentAlgorithms.detection) {
        detectionSelect.value = currentAlgorithms.detection;
    }

    // Tracking algorithm
    const trackingSelect = document.getElementById('trackingAlgorithm');
    if (trackingSelect && currentAlgorithms.tracking) {
        trackingSelect.value = currentAlgorithms.tracking;
    }

    // Loitering algorithm
    const loiteringSelect = document.getElementById('loiteringAlgorithm');
    if (loiteringSelect && currentAlgorithms.loitering) {
        loiteringSelect.value = currentAlgorithms.loitering;
    }

    // Crowd algorithm
    const crowdSelect = document.getElementById('crowdAlgorithm');
    if (crowdSelect && currentAlgorithms.crowd) {
        crowdSelect.value = currentAlgorithms.crowd;
    }

    // Preprocessing algorithm
    const preprocessingSelect = document.getElementById('preprocessingAlgorithm');
    if (preprocessingSelect && currentAlgorithms.preprocessing) {
        preprocessingSelect.value = currentAlgorithms.preprocessing;
    }
}

// Update comparison metrics based on selected component
function updateComparisonMetrics() {
    const componentSelect = document.getElementById('comparisonComponent');
    const metricSelect = document.getElementById('comparisonMetric');

    if (componentSelect && metricSelect) {
        const component = componentSelect.value;
        const metrics = componentMetrics[component] || [];

        // Clear existing options
        metricSelect.innerHTML = '';

        // Add options
        metrics.forEach(metric => {
            const optionElement = document.createElement('option');
            optionElement.value = metric;
            optionElement.textContent = formatMetricName(metric);
            metricSelect.appendChild(optionElement);
        });
    }
}

// Format metric name for display
function formatMetricName(name) {
    // Handle special cases
    const specialCases = {
        'fps': 'FPS',
        'map': 'mAP',
        'mae': 'MAE',
        'mse': 'MSE',
        'mota': 'MOTA',
        'motp': 'MOTP'
    };

    if (specialCases[name]) {
        return specialCases[name];
    }

    // Convert snake_case to Title Case
    return name.split('_').map(word =>
        word.charAt(0).toUpperCase() + word.slice(1)
    ).join(' ');
}

// Initialize charts
function initializeCharts() {
    // Detection charts
    charts.detectionFps = createLineChart('detectionFpsChart', 'FPS', chartColors.blue);
    charts.detectionPrecisionRecall = createMultiLineChart('detectionPrecisionRecallChart',
        ['Precision', 'Recall'], [chartColors.green, chartColors.red]);

    // Tracking charts
    charts.trackingFps = createLineChart('trackingFpsChart', 'FPS', chartColors.blue);
    charts.idSwitches = createLineChart('idSwitchesChart', 'ID Switches', chartColors.orange);

    // Loitering charts
    charts.loiteringEvents = createLineChart('loiteringEventsChart', 'Events', chartColors.purple);
    charts.loiteringPrecisionRecall = createMultiLineChart('loiteringPrecisionRecallChart',
        ['Precision', 'Recall'], [chartColors.green, chartColors.red]);

    // Crowd charts
    charts.crowdEvents = createLineChart('crowdEventsChart', 'Events', chartColors.purple);
    charts.crowdError = createMultiLineChart('crowdErrorChart',
        ['MAE', 'MSE'], [chartColors.yellow, chartColors.orange]);

    // Preprocessing charts
    charts.preprocessingTime = createLineChart('preprocessingTimeChart', 'Processing Time (ms)', chartColors.blue);
    charts.stabilityScore = createLineChart('stabilityScoreChart', 'Stability Score', chartColors.green);

    // Comparison chart is handled by comparison.js
    // charts.comparison = createComparisonChart('comparisonChart');
}

// Create a line chart
function createLineChart(canvasId, label, color) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return null;

    const ctx = canvas.getContext('2d');
    return new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: label,
                data: [],
                backgroundColor: color,
                borderColor: color.replace('0.5', '1'),
                borderWidth: 1,
                pointRadius: 2,
                pointHoverRadius: 4,
                fill: true,
                tension: 0.4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    type: 'time',
                    time: {
                        unit: 'second',
                        displayFormats: {
                            second: 'HH:mm:ss'
                        }
                    },
                    title: {
                        display: true,
                        text: 'Time'
                    }
                },
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: label
                    }
                }
            },
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    mode: 'index',
                    intersect: false
                }
            }
        }
    });
}

// Create a multi-line chart
function createMultiLineChart(canvasId, labels, colors) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return null;

    const ctx = canvas.getContext('2d');

    // Create datasets
    const datasets = labels.map((label, index) => {
        return {
            label: label,
            data: [],
            backgroundColor: colors[index],
            borderColor: colors[index].replace('0.5', '1'),
            borderWidth: 1,
            pointRadius: 2,
            pointHoverRadius: 4,
            fill: false,
            tension: 0.4
        };
    });

    return new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: datasets
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    type: 'time',
                    time: {
                        unit: 'second',
                        displayFormats: {
                            second: 'HH:mm:ss'
                        }
                    },
                    title: {
                        display: true,
                        text: 'Time'
                    }
                },
                y: {
                    beginAtZero: true
                }
            },
            plugins: {
                tooltip: {
                    mode: 'index',
                    intersect: false
                }
            }
        }
    });
}

// Create a comparison chart
function createComparisonChart(canvasId) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return null;

    const ctx = canvas.getContext('2d');
    return new Chart(ctx, {
        type: 'bar',
        data: {
            labels: [],
            datasets: [{
                label: 'Comparison',
                data: [],
                backgroundColor: Object.values(chartColors),
                borderColor: Object.values(chartColors).map(color => color.replace('0.5', '1')),
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true
                }
            },
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    mode: 'index',
                    intersect: false
                }
            }
        }
    });
}

// Set up event listeners
function setupEventListeners() {
    // Disable algorithm selection change events - only use the comparison tab
    // Make all algorithm dropdowns disabled
    document.getElementById('detectionAlgorithm').disabled = true;
    document.getElementById('trackingAlgorithm').disabled = true;
    document.getElementById('loiteringAlgorithm').disabled = true;
    document.getElementById('crowdAlgorithm').disabled = true;
    document.getElementById('preprocessingAlgorithm').disabled = true;

    // Add a note about disabled dropdowns
    const algorithmSelects = document.querySelectorAll('.algorithm-select');
    algorithmSelects.forEach(select => {
        const note = document.createElement('small');
        note.className = 'text-muted d-block mt-1';
        note.textContent = 'Algorithm selection disabled. Use comparison tab to view all algorithms.';
        select.parentNode.appendChild(note);
    });

    // Camera selection change event
    document.getElementById('cameraSelect').addEventListener('change', function() {
        selectedCamera = this.value;
        updateCharts();
    });

    // Comparison component change event
    document.getElementById('comparisonComponent').addEventListener('change', function() {
        updateComparisonMetrics();
        updateComparisonChart();
    });

    // Comparison metric change event
    document.getElementById('comparisonMetric').addEventListener('change', function() {
        updateComparisonChart();
    });

    // Comparison time range change event
    document.getElementById('comparisonTimeRange').addEventListener('change', function() {
        updateComparisonChart();
    });

    // Export data button click event
    document.getElementById('exportDataBtn').addEventListener('click', function() {
        exportData();
    });
}

// Set algorithm for a component
function setAlgorithm(component, algorithm) {
    // Update current algorithms
    currentAlgorithms[component] = algorithm;

    // Send request to server
    fetch('/api/set_algorithm', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            component: component,
            algorithm: algorithm
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            console.log(`Changed ${component} algorithm to ${algorithm}`);
        } else {
            console.error(`Failed to change ${component} algorithm:`, data.error);
        }
    })
    .catch(error => console.error(`Error changing ${component} algorithm:`, error));
}

// Start metrics updates
function startMetricsUpdates() {
    // Request initial metrics
    socket.emit('request_metrics_update', {});

    // Set up periodic updates
    setInterval(function() {
        socket.emit('request_metrics_update', {});
    }, 1000);
}

// Update metrics data
function updateMetricsData(data) {
    if (data.metrics) {
        // Full metrics update
        metricsData = data.metrics;
    } else if (data.component && data.metrics) {
        // Component metrics update
        metricsData[data.component] = data.metrics;
    } else if (data.component && data.metric_name && data.metrics) {
        // Specific metric update
        if (!metricsData[data.component]) {
            metricsData[data.component] = {};
        }

        const key = data.camera_name ? `${data.camera_name}_${data.metric_name}` : data.metric_name;
        metricsData[data.component][key] = data.metrics;
    }
}

// Update charts with current metrics data
function updateCharts() {
    // Update detection charts
    updateDetectionCharts();

    // Update tracking charts
    updateTrackingCharts();

    // Update loitering charts
    updateLoiteringCharts();

    // Update crowd charts
    updateCrowdCharts();

    // Update preprocessing charts
    updatePreprocessingCharts();

    // Update comparison chart
    updateComparisonChart();
}

// Update detection charts
function updateDetectionCharts() {
    if (!metricsData.detection) return;

    // Get metrics for selected camera
    const cameraPrefix = selectedCamera ? `${selectedCamera}_` : '';

    // Update FPS chart
    const fpsData = metricsData.detection[`${cameraPrefix}fps`] || [];
    updateLineChart(charts.detectionFps, fpsData);

    // Update precision/recall chart
    const precisionData = metricsData.detection[`${cameraPrefix}precision`] || [];
    const recallData = metricsData.detection[`${cameraPrefix}recall`] || [];
    updateMultiLineChart(charts.detectionPrecisionRecall, [precisionData, recallData]);
}

// Update tracking charts
function updateTrackingCharts() {
    if (!metricsData.tracking) return;

    // Get metrics for selected camera
    const cameraPrefix = selectedCamera ? `${selectedCamera}_` : '';

    // Update FPS chart
    const fpsData = metricsData.tracking[`${cameraPrefix}fps`] || [];
    updateLineChart(charts.trackingFps, fpsData);

    // Update ID switches chart
    const idSwitchesData = metricsData.tracking[`${cameraPrefix}id_switches`] || [];
    updateLineChart(charts.idSwitches, idSwitchesData);
}

// Update loitering charts
function updateLoiteringCharts() {
    if (!metricsData.loitering) return;

    // Get metrics for selected camera
    const cameraPrefix = selectedCamera ? `${selectedCamera}_` : '';

    // Update events chart
    const eventsData = metricsData.loitering[`${cameraPrefix}event_count`] || [];
    updateLineChart(charts.loiteringEvents, eventsData);

    // Update precision/recall chart
    const precisionData = metricsData.loitering[`${cameraPrefix}precision`] || [];
    const recallData = metricsData.loitering[`${cameraPrefix}recall`] || [];
    updateMultiLineChart(charts.loiteringPrecisionRecall, [precisionData, recallData]);
}

// Update crowd charts
function updateCrowdCharts() {
    if (!metricsData.crowd) return;

    // Get metrics for selected camera
    const cameraPrefix = selectedCamera ? `${selectedCamera}_` : '';

    // Update events chart
    const eventsData = metricsData.crowd[`${cameraPrefix}event_count`] || [];
    updateLineChart(charts.crowdEvents, eventsData);

    // Update error chart
    const maeData = metricsData.crowd[`${cameraPrefix}mae`] || [];
    const mseData = metricsData.crowd[`${cameraPrefix}mse`] || [];
    updateMultiLineChart(charts.crowdError, [maeData, mseData]);
}

// Update preprocessing charts
function updatePreprocessingCharts() {
    if (!metricsData.preprocessing) return;

    // Get metrics for selected camera
    const cameraPrefix = selectedCamera ? `${selectedCamera}_` : '';

    // Update processing time chart
    const timeData = metricsData.preprocessing[`${cameraPrefix}processing_time`] || [];
    updateLineChart(charts.preprocessingTime, timeData);

    // Update stability score chart
    const stabilityData = metricsData.preprocessing[`${cameraPrefix}stability_score`] || [];
    updateLineChart(charts.stabilityScore, stabilityData);
}

// Update comparison metrics dropdown based on selected component
function updateComparisonMetrics() {
    const componentSelect = document.getElementById('comparisonComponent');
    const metricSelect = document.getElementById('comparisonMetric');

    if (!componentSelect || !metricSelect) return;

    const component = componentSelect.value;

    // Clear existing options
    metricSelect.innerHTML = '';

    // Add metrics based on component
    let metrics = [];

    switch (component) {
        case 'detection':
            metrics = ['fps', 'precision', 'recall', 'map'];
            break;
        case 'tracking':
            metrics = ['fps', 'id_switches', 'mota', 'motp'];
            break;
        case 'loitering':
            metrics = ['true_positives', 'false_positives', 'false_negatives', 'precision', 'recall'];
            break;
        case 'crowd':
            metrics = ['mae', 'mse', 'accuracy', 'precision', 'recall', 'event_count'];
            break;
        case 'preprocessing':
            metrics = ['processing_time', 'stability_score'];
            break;
    }

    // Add options to select
    metrics.forEach(metric => {
        const option = document.createElement('option');
        option.value = metric;
        option.textContent = formatMetricName(metric);
        metricSelect.appendChild(option);
    });
}

// Format metric name for display
function formatMetricName(metric) {
    switch (metric) {
        case 'fps': return 'FPS';
        case 'precision': return 'Precision';
        case 'recall': return 'Recall';
        case 'map': return 'mAP';
        case 'id_switches': return 'ID Switches';
        case 'mota': return 'MOTA';
        case 'motp': return 'MOTP';
        case 'true_positives': return 'True Positives';
        case 'false_positives': return 'False Positives';
        case 'false_negatives': return 'False Negatives';
        case 'mae': return 'MAE';
        case 'mse': return 'MSE';
        case 'accuracy': return 'Accuracy';
        case 'event_count': return 'Event Count';
        case 'processing_time': return 'Processing Time (ms)';
        case 'stability_score': return 'Stability Score';
        default: return metric.charAt(0).toUpperCase() + metric.slice(1).replace(/_/g, ' ');
    }
}

// Format algorithm name for display
function formatAlgorithmName(algorithm) {
    switch (algorithm) {
        // Detection algorithms
        case 'background_subtraction': return 'Background Subtraction';
        case 'mog2': return 'MOG2';
        case 'knn': return 'KNN';
        case 'svm_classifier': return 'SVM Classifier';

        // Tracking algorithms
        case 'klt_tracker': return 'KLT Tracker';
        case 'kalman_filter': return 'Kalman Filter';
        case 'optical_flow': return 'Optical Flow';

        // Loitering detection algorithms
        case 'rule_based': return 'Rule-based';
        case 'timer_threshold': return 'Timer Threshold';
        case 'decision_tree': return 'Decision Tree';

        // Crowd detection algorithms
        case 'blob_counting': return 'Blob Counting';
        case 'contour_counting': return 'Contour Counting';
        case 'kmeans_clustering': return 'K-Means Clustering';

        // Preprocessing algorithms
        case 'feature_matching': return 'Feature Matching';
        case 'orb': return 'ORB';
        case 'sift': return 'SIFT';
        case 'affine_transform': return 'Affine Transform';

        // Default formatting
        default: return algorithm.charAt(0).toUpperCase() + algorithm.slice(1).replace(/_/g, ' ');
    }
}

// Update comparison chart - this is now handled by comparison.js
function updateComparisonChart() {
    // This function is intentionally left empty as the comparison chart
    // is now handled by the dedicated comparison.js file

    // We're keeping this function to avoid breaking any existing code that calls it
    console.log("Comparison chart update is now handled by comparison.js");
}

// Update a line chart with new data
function updateLineChart(chart, data) {
    if (!chart) return;

    // Convert data to chart format
    const labels = [];
    const values = [];

    data.forEach(item => {
        labels.push(new Date(item[0] * 1000));
        values.push(item[1]);
    });

    // Update chart
    chart.data.labels = labels;
    chart.data.datasets[0].data = values;
    chart.update();
}

// Update a multi-line chart with new data
function updateMultiLineChart(chart, dataArray) {
    if (!chart) return;

    // Convert data to chart format
    const labels = [];
    const datasets = [];

    // Find common timestamps
    const timestamps = new Set();
    dataArray.forEach(data => {
        data.forEach(item => {
            timestamps.add(item[0]);
        });
    });

    // Sort timestamps
    const sortedTimestamps = Array.from(timestamps).sort();

    // Create labels
    sortedTimestamps.forEach(timestamp => {
        labels.push(new Date(timestamp * 1000));
    });

    // Create datasets
    dataArray.forEach((data, index) => {
        const values = [];

        // Create a map of timestamp to value
        const valueMap = new Map();
        data.forEach(item => {
            valueMap.set(item[0], item[1]);
        });

        // Fill in values for each timestamp
        sortedTimestamps.forEach(timestamp => {
            values.push(valueMap.has(timestamp) ? valueMap.get(timestamp) : null);
        });

        datasets.push(values);
    });

    // Update chart
    chart.data.labels = labels;
    chart.data.datasets.forEach((dataset, index) => {
        if (index < datasets.length) {
            dataset.data = datasets[index];
        }
    });
    chart.update();
}

// Update current performance metrics
function updateCurrentPerformance() {
    // Get metrics for selected camera
    const cameraPrefix = selectedCamera ? `${selectedCamera}_` : '';

    // Detection FPS
    const detectionFps = getLatestMetricValue('detection', `${cameraPrefix}fps`);
    document.getElementById('detectionFps').textContent = detectionFps !== null ? detectionFps.toFixed(1) : '-';

    // Tracking FPS
    const trackingFps = getLatestMetricValue('tracking', `${cameraPrefix}fps`);
    document.getElementById('trackingFps').textContent = trackingFps !== null ? trackingFps.toFixed(1) : '-';

    // Preprocessing time
    const preprocessingTime = getLatestMetricValue('preprocessing', `${cameraPrefix}processing_time`);
    document.getElementById('preprocessingTime').textContent = preprocessingTime !== null ?
        `${preprocessingTime.toFixed(1)} ms` : '-';

    // Stability score
    const stabilityScore = getLatestMetricValue('preprocessing', `${cameraPrefix}stability_score`);
    document.getElementById('stabilityScore').textContent = stabilityScore !== null ?
        stabilityScore.toFixed(2) : '-';
}

// Get the latest value for a metric
function getLatestMetricValue(component, metric) {
    if (!metricsData[component] || !metricsData[component][metric]) {
        return null;
    }

    const data = metricsData[component][metric];
    if (data.length === 0) {
        return null;
    }

    return data[data.length - 1][1];
}

// Export metrics data
function exportData() {
    // Create a JSON blob
    const blob = new Blob([JSON.stringify(metricsData, null, 2)], { type: 'application/json' });

    // Create a download link
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `vigilance_metrics_${new Date().toISOString().replace(/:/g, '-')}.json`;

    // Trigger download
    document.body.appendChild(a);
    a.click();

    // Clean up
    setTimeout(() => {
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }, 0);
}
