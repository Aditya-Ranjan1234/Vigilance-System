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
    
    // Track total detections
    let totalDetections = 0;
    
    // Initialize
    updateStatus();
    
    // Socket.IO event handlers
    socket.on('connect', function() {
        console.log('Connected to server');
    });
    
    socket.on('disconnect', function() {
        console.log('Disconnected from server');
        processingStatus.textContent = 'Disconnected';
        processingStatus.className = 'status-value status-inactive';
    });
    
    socket.on('frame_update', function(data) {
        const cameraName = data.camera;
        const frameBase64 = data.frame;
        const detectionCount = data.detection_count;
        
        // Update camera feed
        const imgElement = document.getElementById(`camera-${cameraName}`);
        if (imgElement) {
            imgElement.src = `data:image/jpeg;base64,${frameBase64}`;
        }
        
        // Update detection count
        const countElement = document.getElementById(`detection-count-${cameraName}`);
        if (countElement) {
            countElement.textContent = detectionCount;
        }
        
        // Update total detections
        totalDetections = 0;
        document.querySelectorAll('.detection-count').forEach(el => {
            totalDetections += parseInt(el.textContent || '0');
        });
        detectionCount.textContent = totalDetections;
    });
    
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
        fetch('/api/start_processing', {
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
        })
        .catch(error => {
            console.error('Error starting processing:', error);
        });
    }
    
    function stopProcessing() {
        fetch('/api/stop_processing', {
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
        })
        .catch(error => {
            console.error('Error stopping processing:', error);
        });
    }
    
    function playAlertSound() {
        // Create audio element
        const audio = new Audio('/static/audio/alert.mp3');
        audio.play().catch(e => {
            console.log('Audio playback failed:', e);
        });
    }
    
    // Refresh status every 5 seconds
    setInterval(updateStatus, 5000);
});
