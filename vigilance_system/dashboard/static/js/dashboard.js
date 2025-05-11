// Dashboard JavaScript

document.addEventListener('DOMContentLoaded', function() {
    // Connect to Socket.IO server
    const socket = io();

    // DOM elements
    const startProcessingBtn = document.getElementById('startProcessingBtn');
    const stopProcessingBtn = document.getElementById('stopProcessingBtn');
    const processingStatus = document.getElementById('processingStatus');
    const cameraCount = document.getElementById('cameraCount');
    const detectionCount = document.getElementById('detectionCount');
    const alertCount = document.getElementById('alertCount');
    const alertList = document.getElementById('alertList');

    // Camera view mode controls
    const viewModeGrid = document.getElementById('viewModeGrid');
    const viewModeSingle = document.getElementById('viewModeSingle');
    const cameraGrid = document.getElementById('cameraGrid');
    const cameraCols = document.querySelectorAll('.camera-col');

    // Current active camera in single view mode
    let activeCameraIndex = 0;

    // Algorithm selection controls
    const algorithmSelector = document.getElementById('algorithmSelector');
    const trackingSelector = document.getElementById('trackingSelector');
    const classifierSelector = document.getElementById('classifierSelector');
    const analysisSelector = document.getElementById('analysisSelector');

    // Network controls
    const frameRateSelector = document.getElementById('frameRateSelector');
    const resolutionSelector = document.getElementById('resolutionSelector');
    const routingSelector = document.getElementById('routingSelector');

    // Network metrics displays
    const bandwidthDisplay = document.getElementById('bandwidthDisplay');
    const latencyDisplay = document.getElementById('latencyDisplay');
    const packetLossDisplay = document.getElementById('packetLossDisplay');
    const jitterDisplay = document.getElementById('jitterDisplay');

    const applyAlgorithmsBtn = document.getElementById('applyAlgorithmsBtn');

    // Algorithm visualization controls
    const showStabilization = document.getElementById('showStabilization');
    const showTracking = document.getElementById('showTracking');
    const showDecisionMaking = document.getElementById('showDecisionMaking');
    const showAlgorithmSteps = document.getElementById('showAlgorithmSteps');
    const applyVisualizationsBtn = document.getElementById('applyVisualizationsBtn');

    // Track total detections
    let totalDetections = 0;

    // Flag to track if user has made changes to algorithm selectors
    let algorithmSelectorsChanged = false;

    // Add change event listeners to algorithm selectors to track user changes
    algorithmSelector.addEventListener('change', function() {
        algorithmSelectorsChanged = true;
        // Highlight the apply button to indicate changes need to be applied
        applyAlgorithmsBtn.classList.add('btn-warning');
    });

    trackingSelector.addEventListener('change', function() {
        algorithmSelectorsChanged = true;
        applyAlgorithmsBtn.classList.add('btn-warning');
    });

    classifierSelector.addEventListener('change', function() {
        algorithmSelectorsChanged = true;
        applyAlgorithmsBtn.classList.add('btn-warning');
    });

    analysisSelector.addEventListener('change', function() {
        algorithmSelectorsChanged = true;
        applyAlgorithmsBtn.classList.add('btn-warning');
    });

    // Initialize
    updateStatus();

    // Mark non-working algorithms as disabled
    markNonWorkingAlgorithms();

    // Socket.IO event handlers
    socket.on('connect', function() {
        console.log('Connected to server');
    });

    socket.on('disconnect', function() {
        console.log('Disconnected from server');
        processingStatus.textContent = 'Disconnected';
        processingStatus.className = 'status-value status-inactive';
    });

    // Track frame timestamps for frame rate calculation
    let frameTimestamps = {};

    // Direct frame update without queuing for more reliable playback
    socket.on('frame_update', function(data) {
        const cameraName = data.camera;
        const frameBase64 = data.frame;
        const detectionCount = data.detection_count;
        const currentAlgorithms = data.current_algorithms;

        // Update camera feed directly
        const imgElement = document.getElementById(`camera-${cameraName}`);
        if (imgElement) {
            imgElement.src = `data:image/jpeg;base64,${frameBase64}`;

            // Add error handling for image loading
            imgElement.onerror = function() {
                console.error(`Error loading frame for camera ${cameraName}`);
                // Try to recover by setting a placeholder
                setTimeout(() => {
                    imgElement.src = '/static/img/camera-placeholder.jpg';
                }, 500);
            };
        }

        // Update frame rate
        updateFrameRate(cameraName);

        // Update detection count (minimal UI in video feed)
        const countElement = document.getElementById(`detection-count-${cameraName}`);
        if (countElement) {
            countElement.textContent = detectionCount;
        }

        // Update total detections
        totalDetections = 0;
        document.querySelectorAll('.detection-count').forEach(el => {
            totalDetections += parseInt(el.textContent || '0');
        });

        // Make sure detectionCount element exists before updating
        if (detectionCount) {
            detectionCount.textContent = totalDetections;
        }

        // Update algorithm selectors to match what's actually being used
        if (currentAlgorithms) {
            // Only update if the user hasn't made any changes to the algorithm selectors
            // and the apply button isn't currently disabled (indicating an update in progress)
            if (!algorithmSelectorsChanged && !applyAlgorithmsBtn.disabled) {
                // Store current values to check if they actually change
                const prevDetection = algorithmSelector.value;
                const prevTracking = trackingSelector.value;
                const prevClassifier = classifierSelector.value;
                const prevAnalysis = analysisSelector.value;

                // Only update if values are different
                if (currentAlgorithms.detection && prevDetection !== currentAlgorithms.detection) {
                    algorithmSelector.value = currentAlgorithms.detection;
                }
                if (currentAlgorithms.tracking && prevTracking !== currentAlgorithms.tracking) {
                    trackingSelector.value = currentAlgorithms.tracking;
                }
                if (currentAlgorithms.classification && prevClassifier !== currentAlgorithms.classification) {
                    classifierSelector.value = currentAlgorithms.classification;
                }
                if (currentAlgorithms.analysis && prevAnalysis !== currentAlgorithms.analysis) {
                    analysisSelector.value = currentAlgorithms.analysis;
                }
            }
        }
    });

    // Function to monitor frame rate
    function updateFrameRate(cameraName) {
        if (!frameTimestamps[cameraName]) {
            frameTimestamps[cameraName] = [];
        }

        const now = Date.now();
        frameTimestamps[cameraName].push(now);

        // Keep only the last 10 timestamps
        if (frameTimestamps[cameraName].length > 10) {
            frameTimestamps[cameraName].shift();
        }

        // Calculate FPS if we have enough data
        if (frameTimestamps[cameraName].length >= 2) {
            const elapsed = now - frameTimestamps[cameraName][0];
            const fps = ((frameTimestamps[cameraName].length - 1) * 1000 / elapsed).toFixed(1);

            // Update FPS display if available
            const fpsElement = document.getElementById(`fps-${cameraName}`);
            if (fpsElement) {
                fpsElement.textContent = `${fps} FPS`;
            }
        }
    }

    socket.on('new_alert', function(alert) {
        // Add alert to list
        addAlertToList(alert);

        // Update alert count
        const currentCount = parseInt(alertCount.textContent || '0');
        alertCount.textContent = currentCount + 1;

        // Play notification sound
        playAlertSound();
    });

    // Button event handlers
    startProcessingBtn.addEventListener('click', function(e) {
        e.preventDefault();
        startProcessing();
    });

    stopProcessingBtn.addEventListener('click', function(e) {
        e.preventDefault();
        stopProcessing();
    });

    applyAlgorithmsBtn.addEventListener('click', function(e) {
        e.preventDefault();
        updateAlgorithmSettings();
    });

    applyVisualizationsBtn.addEventListener('click', function(e) {
        e.preventDefault();
        updateVisualizationSettings();
    });

    // View mode event handlers
    viewModeGrid.addEventListener('click', function(e) {
        e.preventDefault();
        setViewMode('grid');
    });

    viewModeSingle.addEventListener('click', function(e) {
        e.preventDefault();
        setViewMode('single');
    });

    // Add click event to camera containers to switch to single view
    document.querySelectorAll('.camera-container').forEach((container, index) => {
        container.addEventListener('click', function() {
            activeCameraIndex = index;
            setViewMode('single');
        });
    });

    // Functions
    function updateStatus() {
        fetch('/api/status')
            .then(response => response.json())
            .then(data => {
                // Update processing status
                if (data.processing_active) {
                    processingStatus.textContent = 'Active';
                    processingStatus.className = 'status-value status-active';
                } else {
                    processingStatus.textContent = 'Inactive';
                    processingStatus.className = 'status-value status-inactive';
                }

                // Update camera count
                cameraCount.textContent = data.camera_count;

                // Update detection stats
                detectionCount.textContent = data.detection_stats.total_detections || 0;

                // Update alert count
                alertCount.textContent = data.alert_count;

                // Update visualization controls
                if (data.visualizations) {
                    showStabilization.checked = data.visualizations.show_stabilization;
                    showTracking.checked = data.visualizations.show_tracking;
                    showDecisionMaking.checked = data.visualizations.show_decision_making;
                    if (data.visualizations.show_algorithm_steps !== undefined) {
                        showAlgorithmSteps.checked = data.visualizations.show_algorithm_steps;
                    }
                }

                // Update algorithm selectors
                if (data.algorithms) {
                    if (data.algorithms.detection_algorithm) {
                        algorithmSelector.value = data.algorithms.detection_algorithm;
                    }
                    if (data.algorithms.tracking_algorithm) {
                        trackingSelector.value = data.algorithms.tracking_algorithm;
                    }
                    if (data.algorithms.classifier_algorithm) {
                        classifierSelector.value = data.algorithms.classifier_algorithm;
                    }
                    if (data.algorithms.analysis_algorithm) {
                        analysisSelector.value = data.algorithms.analysis_algorithm;
                    }
                }

                // Update network settings
                if (data.network) {
                    if (data.network.frame_rate) {
                        frameRateSelector.value = data.network.frame_rate;
                    }
                    if (data.network.resolution) {
                        resolutionSelector.value = data.network.resolution;
                    }
                    if (data.network.routing_algorithm) {
                        routingSelector.value = data.network.routing_algorithm;
                    }

                    // Update network metrics display
                    if (data.network.metrics) {
                        bandwidthDisplay.value = data.network.metrics.bandwidth;
                        latencyDisplay.value = data.network.metrics.latency;
                        packetLossDisplay.value = data.network.metrics.packet_loss;
                        jitterDisplay.value = data.network.metrics.jitter;
                    } else {
                        // Calculate metrics based on current settings
                        updateNetworkMetrics({
                            frame_rate: parseInt(frameRateSelector.value),
                            resolution: resolutionSelector.value,
                            routing_algorithm: routingSelector.value
                        });
                    }
                }
            })
            .catch(error => {
                console.error('Error fetching status:', error);
            });

        // Update alerts
        updateAlerts();
    }

    function updateAlerts() {
        fetch('/api/alerts')
            .then(response => response.json())
            .then(data => {
                // Clear current alerts
                alertList.innerHTML = '';

                // Add alerts to list
                data.alerts.forEach(alert => {
                    addAlertToList(alert);
                });
            })
            .catch(error => {
                console.error('Error fetching alerts:', error);
            });
    }

    function addAlertToList(alert) {
        const alertItem = document.createElement('div');
        alertItem.className = 'alert-item';

        const alertTime = new Date(alert.timestamp * 1000).toLocaleTimeString();
        const alertType = alert.type;
        const alertMessage = alert.message;

        alertItem.innerHTML = `
            <div class="alert-time">${alertTime}</div>
            <div class="alert-message">${alertMessage}</div>
            <span class="alert-type alert-type-${alertType}">${alertType}</span>
        `;

        // Add to beginning of list
        alertList.insertBefore(alertItem, alertList.firstChild);

        // Limit number of alerts shown
        if (alertList.children.length > 20) {
            alertList.removeChild(alertList.lastChild);
        }
    }

    function startProcessing() {
        return fetch('/api/start_processing', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        })
        .then(response => response.json())
        .then(data => {
            if (data.status === 'started' || data.status === 'already_running') {
                processingStatus.textContent = 'Active';
                processingStatus.className = 'status-value status-active';
            }
            return data;
        })
        .catch(error => {
            console.error('Error starting processing:', error);
            throw error;
        });
    }

    function stopProcessing() {
        return fetch('/api/stop_processing', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        })
        .then(response => response.json())
        .then(data => {
            if (data.status === 'stopping' || data.status === 'not_running') {
                processingStatus.textContent = 'Inactive';
                processingStatus.className = 'status-value status-inactive';
            }
            return data;
        })
        .catch(error => {
            console.error('Error stopping processing:', error);
            throw error;
        });
    }

    function playAlertSound() {
        // Create audio element
        const audio = new Audio('/static/audio/alert.mp3');
        audio.play().catch(e => {
            console.log('Audio playback failed:', e);
        });
    }

    function markNonWorkingAlgorithms() {
        // Define which algorithms are working and which are not
        const workingAlgorithms = {
            detection: ['yolov5s', 'yolov5m', 'yolov5l', 'background_subtraction', 'hog_svm'],
            tracking: ['centroid', 'knn', 'svm', 'iou', 'kalman', 'naive_bayes', 'decision_tree', 'random_forest'],
            classifier: ['svm', 'knn', 'naive_bayes', 'decision_tree', 'random_forest'],
            analysis: ['basic', 'crowd']
        };

        // Mark non-working detection algorithms
        Array.from(algorithmSelector.options).forEach(option => {
            if (!workingAlgorithms.detection.includes(option.value)) {
                option.classList.add('algorithm-option-disabled');
                option.disabled = true;
            }
        });

        // Mark non-working tracking algorithms
        Array.from(trackingSelector.options).forEach(option => {
            if (!workingAlgorithms.tracking.includes(option.value)) {
                option.classList.add('algorithm-option-disabled');
                option.disabled = true;
            }
        });

        // Mark non-working classifier algorithms
        Array.from(classifierSelector.options).forEach(option => {
            if (!workingAlgorithms.classifier.includes(option.value)) {
                option.classList.add('algorithm-option-disabled');
                option.disabled = true;
            }
        });

        // Mark non-working analysis algorithms
        Array.from(analysisSelector.options).forEach(option => {
            if (!workingAlgorithms.analysis.includes(option.value)) {
                option.classList.add('algorithm-option-disabled');
                option.disabled = true;
            }
        });
    }

    function updateAlgorithmSettings() {
        // Show loading indicator
        const btn = applyAlgorithmsBtn;
        const originalText = btn.textContent;
        btn.textContent = 'Applying Changes...';
        btn.disabled = true;

        // Highlight the algorithm selectors to show they're being updated
        algorithmSelector.classList.add('border-primary');
        trackingSelector.classList.add('border-primary');
        classifierSelector.classList.add('border-primary');
        analysisSelector.classList.add('border-primary');

        // Store the selected values to verify they're applied correctly
        const selectedDetection = algorithmSelector.value;
        const selectedTracking = trackingSelector.value;
        const selectedClassifier = classifierSelector.value;
        const selectedAnalysis = analysisSelector.value;

        // Get current algorithm settings
        const settings = {
            detection_algorithm: selectedDetection,
            tracking_algorithm: selectedTracking,
            classifier_algorithm: selectedClassifier,
            analysis_algorithm: selectedAnalysis,
            network: {
                frame_rate: parseInt(frameRateSelector.value),
                resolution: resolutionSelector.value,
                routing_algorithm: routingSelector.value
            }
        };

        // Create visual feedback for algorithm change
        document.querySelectorAll('.camera-container').forEach(container => {
            // Add a temporary overlay to indicate processing
            const overlay = document.createElement('div');
            overlay.className = 'algorithm-change-overlay';
            overlay.innerHTML = `
                <div class="algorithm-change-message">
                    <div class="spinner-border text-light spinner-border-sm me-2" role="status">
                        <span class="visually-hidden">Loading...</span>
                    </div>
                    Applying Algorithm Changes...
                </div>
            `;
            container.appendChild(overlay);
        });

        // Send settings to server
        fetch('/api/update_algorithms', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(settings)
        })
        .then(response => response.json())
        .then(data => {
            if (data.status === 'success') {
                // Show success message
                btn.textContent = 'Settings Updated!';
                btn.classList.add('btn-success');
                btn.classList.remove('btn-sm');

                // Update network metrics based on new settings
                updateNetworkMetrics(settings.network);

                // Verify the settings were applied by fetching the current status
                fetch('/api/status')
                    .then(response => response.json())
                    .then(statusData => {
                        // Check if the algorithms match what we set
                        if (statusData.algorithms) {
                            const currentDetection = statusData.algorithms.detection_algorithm;
                            const currentTracking = statusData.algorithms.tracking_algorithm;
                            const currentClassifier = statusData.algorithms.classifier_algorithm;
                            const currentAnalysis = statusData.algorithms.analysis_algorithm;

                            // Log any mismatches for debugging
                            if (currentDetection !== selectedDetection) {
                                console.warn(`Detection algorithm mismatch: set ${selectedDetection}, got ${currentDetection}`);
                            }
                            if (currentTracking !== selectedTracking) {
                                console.warn(`Tracking algorithm mismatch: set ${selectedTracking}, got ${currentTracking}`);
                            }
                            if (currentClassifier !== selectedClassifier) {
                                console.warn(`Classifier algorithm mismatch: set ${selectedClassifier}, got ${currentClassifier}`);
                            }
                            if (currentAnalysis !== selectedAnalysis) {
                                console.warn(`Analysis algorithm mismatch: set ${selectedAnalysis}, got ${currentAnalysis}`);
                            }

                            // Update the selectors to match the actual values
                            algorithmSelector.value = currentDetection;
                            trackingSelector.value = currentTracking;
                            classifierSelector.value = currentClassifier;
                            analysisSelector.value = currentAnalysis;
                        }
                    })
                    .catch(error => {
                        console.error('Error verifying algorithm settings:', error);
                    });

                // Remove the algorithm change overlays after a delay
                setTimeout(() => {
                    document.querySelectorAll('.algorithm-change-overlay').forEach(overlay => {
                        overlay.remove();
                    });

                    // Reset button
                    btn.textContent = originalText;
                    btn.disabled = false;
                    btn.classList.remove('btn-success');
                    btn.classList.remove('btn-warning');
                    btn.classList.add('btn-sm');

                    // Remove highlights
                    algorithmSelector.classList.remove('border-primary');
                    trackingSelector.classList.remove('border-primary');
                    classifierSelector.classList.remove('border-primary');
                    analysisSelector.classList.remove('border-primary');

                    // Reset the change flag since changes have been applied
                    algorithmSelectorsChanged = false;
                }, 2000);

                // Restart processing if it was active
                if (processingStatus.textContent === 'Active') {
                    stopProcessing().then(() => {
                        setTimeout(() => {
                            startProcessing();
                        }, 1000);
                    });
                }
            } else {
                // Show error
                btn.textContent = 'Error!';
                btn.classList.add('btn-danger');

                // Reset after delay
                setTimeout(() => {
                    btn.textContent = originalText;
                    btn.disabled = false;
                    btn.classList.remove('btn-danger');
                    // Keep the warning state since changes weren't applied
                    btn.classList.add('btn-warning');

                    // Remove overlays
                    document.querySelectorAll('.algorithm-change-overlay').forEach(overlay => {
                        overlay.remove();
                    });

                    // Remove highlights
                    algorithmSelector.classList.remove('border-primary');
                    trackingSelector.classList.remove('border-primary');
                    classifierSelector.classList.remove('border-primary');
                    analysisSelector.classList.remove('border-primary');

                    // Keep algorithmSelectorsChanged flag as true since changes weren't applied
                }, 2000);
            }
        })
        .catch(error => {
            console.error('Error updating algorithm settings:', error);

            // Show error
            btn.textContent = 'Error!';
            btn.classList.add('btn-danger');

            // Reset after delay
            setTimeout(() => {
                btn.textContent = originalText;
                btn.disabled = false;
                btn.classList.remove('btn-danger');
                // Keep the warning state since changes weren't applied
                btn.classList.add('btn-warning');

                // Remove overlays
                document.querySelectorAll('.algorithm-change-overlay').forEach(overlay => {
                    overlay.remove();
                });

                // Remove highlights
                algorithmSelector.classList.remove('border-primary');
                trackingSelector.classList.remove('border-primary');
                classifierSelector.classList.remove('border-primary');
                analysisSelector.classList.remove('border-primary');

                // Keep algorithmSelectorsChanged flag as true since changes weren't applied
            }, 2000);
        });
    }

    function updateNetworkMetrics(networkSettings) {
        // Fetch real metrics from the server
        fetch('/api/status')
            .then(response => response.json())
            .then(data => {
                if (data.network && data.network.metrics) {
                    // Use real metrics from the network simulator
                    bandwidthDisplay.value = data.network.metrics.bandwidth;
                    latencyDisplay.value = data.network.metrics.latency;
                    packetLossDisplay.value = data.network.metrics.packet_loss;
                    jitterDisplay.value = data.network.metrics.jitter;
                } else {
                    // Fallback to calculated metrics if real ones aren't available
                    // Calculate estimated bandwidth based on resolution and frame rate
                    let bandwidth = 0;
                    if (networkSettings.resolution === 'low') {
                        bandwidth = networkSettings.frame_rate * 0.1; // 0.1 Mbps per frame at 480p
                    } else if (networkSettings.resolution === 'medium') {
                        bandwidth = networkSettings.frame_rate * 0.2; // 0.2 Mbps per frame at 720p
                    } else if (networkSettings.resolution === 'high') {
                        bandwidth = networkSettings.frame_rate * 0.4; // 0.4 Mbps per frame at 1080p
                    }

                    // Add some randomness to make it look realistic
                    const randomFactor = 0.9 + Math.random() * 0.2; // 0.9 to 1.1
                    bandwidth = (bandwidth * randomFactor).toFixed(1);

                    // Calculate latency based on routing algorithm and bandwidth
                    let latency = 30; // Base latency in ms
                    if (networkSettings.routing_algorithm === 'direct') {
                        latency += 10;
                    } else if (networkSettings.routing_algorithm === 'round_robin') {
                        latency += 15;
                    } else if (networkSettings.routing_algorithm === 'least_connection') {
                        latency += 20;
                    } else if (networkSettings.routing_algorithm === 'weighted') {
                        latency += 25;
                    } else if (networkSettings.routing_algorithm === 'ip_hash') {
                        latency += 30;
                    }

                    // Add some randomness to latency
                    latency = Math.round(latency * (0.9 + Math.random() * 0.2));

                    // Calculate packet loss (very low for most cases)
                    const packetLoss = (0.05 + Math.random() * 0.1).toFixed(2);

                    // Calculate jitter (variation in latency)
                    const jitter = Math.round(latency * 0.2 + Math.random() * 5);

                    // Update the display elements
                    bandwidthDisplay.value = bandwidth + " Mbps";
                    latencyDisplay.value = latency + " ms";
                    packetLossDisplay.value = packetLoss + "%";
                    jitterDisplay.value = jitter + " ms";
                }
            })
            .catch(error => {
                console.error('Error fetching network metrics:', error);
            });
    }

    function updateVisualizationSettings() {
        // Get current visualization settings
        const settings = {
            show_stabilization: showStabilization.checked,
            show_tracking: showTracking.checked,
            show_decision_making: showDecisionMaking.checked,
            show_algorithm_steps: showAlgorithmSteps.checked
        };

        // Send settings to server
        fetch('/api/update_visualizations', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(settings)
        })
        .then(response => response.json())
        .then(data => {
            if (data.status === 'success') {
                // Show success message
                const btn = applyVisualizationsBtn;
                const originalText = btn.textContent;

                btn.textContent = 'Settings Applied!';
                btn.classList.add('btn-success');
                btn.classList.remove('btn-primary');

                // Reset button after 2 seconds
                setTimeout(() => {
                    btn.textContent = originalText;
                    btn.classList.add('btn-primary');
                    btn.classList.remove('btn-success');
                }, 2000);
            }
        })
        .catch(error => {
            console.error('Error updating visualization settings:', error);
        });
    }

    // View mode functions
    function setViewMode(mode) {
        const cameras = document.querySelectorAll('.camera-col');

        if (mode === 'grid') {
            // Reset all cameras to grid view
            cameras.forEach(camera => {
                camera.classList.remove('d-none');
                camera.classList.add('col-md-6');
                camera.classList.remove('col-md-12');
                camera.querySelector('.camera-container').classList.remove('camera-fullscreen');
            });

            // Hide decision tree visualization
            document.getElementById('decision-tree-container').classList.add('d-none');

            // Update button states
            viewModeGrid.classList.add('active');
            viewModeSingle.classList.remove('active');

            // Tell server we're in grid mode
            fetch('/api/update_visualizations', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    view_mode: 'grid',
                    show_stabilization: showStabilization.checked,
                    show_tracking: showTracking.checked,
                    show_decision_making: showDecisionMaking.checked,
                    show_algorithm_steps: showAlgorithmSteps.checked
                })
            });

        } else if (mode === 'single') {
            // Hide all cameras except the active one
            cameras.forEach((camera, index) => {
                if (index === activeCameraIndex) {
                    camera.classList.remove('d-none');
                    camera.classList.remove('col-md-6');
                    camera.classList.add('col-md-12');
                    camera.querySelector('.camera-container').classList.add('camera-fullscreen');

                    // Show decision tree visualization for this camera
                    const cameraName = camera.querySelector('.camera-title').textContent.trim();
                    showDecisionTree(cameraName);
                } else {
                    camera.classList.add('d-none');
                }
            });

            // Update button states
            viewModeGrid.classList.remove('active');
            viewModeSingle.classList.add('active');

            // Tell server we're in single mode
            fetch('/api/update_visualizations', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    view_mode: 'single',
                    show_stabilization: showStabilization.checked,
                    show_tracking: showTracking.checked,
                    show_decision_making: showDecisionMaking.checked,
                    show_algorithm_steps: showAlgorithmSteps.checked
                })
            });
        }
    }

    // Function to show decision tree visualization
    function showDecisionTree(cameraName) {
        // Get the decision tree container
        const container = document.getElementById('decision-tree-container');

        // Show the container
        container.classList.remove('d-none');

        // Get current algorithm settings
        const detectionAlgo = algorithmSelector.value;
        const trackingAlgo = trackingSelector.value;
        const classifierAlgo = classifierSelector.value;
        const analysisAlgo = analysisSelector.value;

        // Create decision tree visualization
        let decisionTreeHtml = `
            <h5 class="mb-3">Decision Making Process for ${cameraName}</h5>
            <div class="decision-tree">
                <div class="tree-node root">
                    <div class="node-content">
                        <div class="node-title">Input Frame</div>
                        <div class="node-desc">Raw video frame from camera</div>
                    </div>
                    <div class="tree-connector"></div>
                </div>

                <div class="tree-node">
                    <div class="node-content">
                        <div class="node-title">Detection (${detectionAlgo})</div>
                        <div class="node-desc">Identifies objects in frame</div>
                        <div class="node-threshold">Threshold: ${showStabilization.checked ? '0.5' : '0.4'}</div>
                    </div>
                    <div class="tree-connector"></div>
                </div>

                <div class="tree-node">
                    <div class="node-content">
                        <div class="node-title">Tracking (${trackingAlgo})</div>
                        <div class="node-desc">Maintains object identity across frames</div>
                        <div class="node-threshold">Max Distance: 50px</div>
                    </div>
                    <div class="tree-connector"></div>
                </div>

                <div class="tree-node">
                    <div class="node-content">
                        <div class="node-title">Classification (${classifierAlgo})</div>
                        <div class="node-desc">Categorizes detected objects</div>
                        <div class="node-threshold">Confidence: 0.7</div>
                    </div>
                    <div class="tree-connector"></div>
                </div>

                <div class="tree-node">
                    <div class="node-content">
                        <div class="node-title">Analysis (${analysisAlgo})</div>
                        <div class="node-desc">Evaluates situation based on rules</div>
                    </div>
                    <div class="tree-connector"></div>
                </div>

                <div class="tree-node decision">
                    <div class="node-content">
                        <div class="node-title">Decision Rules</div>
                        <div class="decision-rules">
                            <div class="rule">
                                <span class="rule-name">Loitering:</span>
                                <span class="rule-desc">Person present > 30s</span>
                            </div>
                            <div class="rule">
                                <span class="rule-name">Crowd:</span>
                                <span class="rule-desc">People count > 3</span>
                            </div>
                            <div class="rule">
                                <span class="rule-name">Restricted Area:</span>
                                <span class="rule-desc">Person in defined zone</span>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        `;

        // Set the HTML
        container.innerHTML = decisionTreeHtml;
    }

    // Initialize with grid view
    viewModeGrid.classList.add('active');

    // Refresh status every 5 seconds
    setInterval(updateStatus, 5000);
});
