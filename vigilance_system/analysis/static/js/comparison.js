// Comparison chart functionality

// Sample data for algorithm comparison
const sampleData = {
    detection: {
        fps: {
            background_subtraction: 45.0,
            mog2: 40.0,
            knn: 38.0,
            svm_classifier: 35.0
        },
        precision: {
            background_subtraction: 0.70,
            mog2: 0.75,
            knn: 0.72,
            svm_classifier: 0.78
        },
        recall: {
            background_subtraction: 0.65,
            mog2: 0.70,
            knn: 0.68,
            svm_classifier: 0.72
        },
        map: {
            background_subtraction: 0.60,
            mog2: 0.65,
            knn: 0.62,
            svm_classifier: 0.68
        }
    },
    tracking: {
        fps: {
            klt_tracker: 42.0,
            kalman_filter: 38.0,
            optical_flow: 35.0
        },
        id_switches: {
            klt_tracker: 8.0,
            kalman_filter: 6.0,
            optical_flow: 7.0
        },
        mota: {
            klt_tracker: 0.65,
            kalman_filter: 0.70,
            optical_flow: 0.68
        },
        motp: {
            klt_tracker: 0.68,
            kalman_filter: 0.72,
            optical_flow: 0.70
        }
    },
    loitering: {
        true_positives: {
            rule_based: 7.0,
            timer_threshold: 8.0,
            decision_tree: 7.5
        },
        false_positives: {
            rule_based: 4.0,
            timer_threshold: 3.5,
            decision_tree: 3.0
        },
        false_negatives: {
            rule_based: 3.0,
            timer_threshold: 2.5,
            decision_tree: 2.8
        },
        precision: {
            rule_based: 0.65,
            timer_threshold: 0.70,
            decision_tree: 0.72
        },
        recall: {
            rule_based: 0.70,
            timer_threshold: 0.75,
            decision_tree: 0.73
        }
    },
    crowd: {
        mae: {
            blob_counting: 3.0,
            contour_counting: 2.8,
            kmeans_clustering: 2.5
        },
        mse: {
            blob_counting: 10.0,
            contour_counting: 9.0,
            kmeans_clustering: 8.0
        },
        accuracy: {
            blob_counting: 0.68,
            contour_counting: 0.70,
            kmeans_clustering: 0.72
        },
        precision: {
            blob_counting: 0.65,
            contour_counting: 0.68,
            kmeans_clustering: 0.70
        },
        recall: {
            blob_counting: 0.62,
            contour_counting: 0.65,
            kmeans_clustering: 0.68
        },
        event_count: {
            blob_counting: 2.5,
            contour_counting: 3.0,
            kmeans_clustering: 3.2
        }
    },
    preprocessing: {
        processing_time: {
            feature_matching: 10.0,
            orb: 8.0,
            sift: 12.0,
            affine_transform: 9.0
        },
        stability_score: {
            feature_matching: 0.72,
            orb: 0.70,
            sift: 0.75,
            affine_transform: 0.78
        }
    }
};

// Available algorithms for each component
const availableAlgorithms = {
    detection: ['background_subtraction', 'mog2', 'knn', 'svm_classifier'],
    tracking: ['klt_tracker', 'kalman_filter', 'optical_flow'],
    loitering: ['rule_based', 'timer_threshold', 'decision_tree'],
    crowd: ['blob_counting', 'contour_counting', 'kmeans_clustering'],
    preprocessing: ['feature_matching', 'orb', 'sift', 'affine_transform']
};

// Available metrics for each component
const availableMetrics = {
    detection: ['fps', 'precision', 'recall', 'map'],
    tracking: ['fps', 'id_switches', 'mota', 'motp'],
    loitering: ['true_positives', 'false_positives', 'false_negatives', 'precision', 'recall'],
    crowd: ['mae', 'mse', 'accuracy', 'precision', 'recall', 'event_count'],
    preprocessing: ['processing_time', 'stability_score']
};

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

// Initialize comparison chart
let comparisonChart = null;

function initializeComparisonChart() {
    const canvas = document.getElementById('comparisonChart');
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    comparisonChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: [],
            datasets: [{
                label: 'Comparison',
                data: [],
                backgroundColor: [
                    'rgba(75, 192, 192, 0.6)',
                    'rgba(54, 162, 235, 0.6)',
                    'rgba(255, 206, 86, 0.6)',
                    'rgba(255, 99, 132, 0.6)',
                    'rgba(153, 102, 255, 0.6)'
                ],
                borderColor: [
                    'rgba(75, 192, 192, 1)',
                    'rgba(54, 162, 235, 1)',
                    'rgba(255, 206, 86, 1)',
                    'rgba(255, 99, 132, 1)',
                    'rgba(153, 102, 255, 1)'
                ],
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Value'
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

    // Update comparison metrics dropdown
    updateComparisonMetrics();

    // Update comparison chart
    updateComparisonChart();
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
    const metrics = availableMetrics[component] || [];

    // Add options to select
    metrics.forEach(metric => {
        const option = document.createElement('option');
        option.value = metric;
        option.textContent = formatMetricName(metric);
        metricSelect.appendChild(option);
    });
}

// Update comparison chart
function updateComparisonChart() {
    const componentSelect = document.getElementById('comparisonComponent');
    const metricSelect = document.getElementById('comparisonMetric');

    if (!componentSelect || !metricSelect || !comparisonChart) return;

    const component = componentSelect.value;
    const metric = metricSelect.value;

    if (!component || !metric || !sampleData[component] || !sampleData[component][metric]) return;

    // Get data for the selected component and metric
    const metricData = sampleData[component][metric];

    // Prepare data for chart
    const labels = [];
    const data = [];
    const backgroundColors = [
        'rgba(75, 192, 192, 0.6)',
        'rgba(54, 162, 235, 0.6)',
        'rgba(255, 206, 86, 0.6)',
        'rgba(255, 99, 132, 0.6)',
        'rgba(153, 102, 255, 0.6)'
    ];

    // Add data for each algorithm
    Object.entries(metricData).forEach(([algorithm, value], index) => {
        labels.push(formatAlgorithmName(algorithm));
        data.push(value);
    });

    // Update chart
    comparisonChart.data.labels = labels;
    comparisonChart.data.datasets[0].data = data;
    comparisonChart.data.datasets[0].label = formatMetricName(metric);
    comparisonChart.data.datasets[0].backgroundColor = backgroundColors.slice(0, labels.length);
    comparisonChart.options.scales.y.title = {
        display: true,
        text: formatMetricName(metric)
    };
    comparisonChart.update();
}

// Set up event listeners
function setupComparisonEventListeners() {
    // Comparison component change event
    const componentSelect = document.getElementById('comparisonComponent');
    if (componentSelect) {
        componentSelect.addEventListener('change', function() {
            updateComparisonMetrics();
            updateComparisonChart();
        });
    }

    // Comparison metric change event
    const metricSelect = document.getElementById('comparisonMetric');
    if (metricSelect) {
        metricSelect.addEventListener('change', function() {
            updateComparisonChart();
        });
    }
}

// Initialize when the DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    // Wait a short time to ensure the dashboard.js has finished initializing
    setTimeout(function() {
        initializeComparisonChart();
        setupComparisonEventListeners();

        // Force an update of the comparison chart
        updateComparisonChart();

        console.log("Comparison chart initialized");
    }, 500);
});
