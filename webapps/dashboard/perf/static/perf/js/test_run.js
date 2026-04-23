function showStatus(message, type) {
    const statusDiv = document.getElementById('status');
    const alertType = type === 'error' ? 'danger' : type;

    statusDiv.innerHTML = '';

    const messageSpan = document.createElement('span');
    messageSpan.textContent = message;

    const dismissButton = document.createElement('button');
    dismissButton.type = 'button';
    dismissButton.className = 'btn-close';
    dismissButton.setAttribute('aria-label', 'Close');
    dismissButton.addEventListener('click', () => {
        statusDiv.style.display = 'none';
    });

    statusDiv.append(messageSpan, dismissButton);
    statusDiv.className = `alert alert-${alertType} alert-dismissible align-items-center justify-content-between gap-2`;
    statusDiv.style.display = 'block';
}

function initializeTooltips() {
    document.querySelectorAll('[data-bs-toggle="tooltip"]').forEach(element => {
        new bootstrap.Tooltip(element, {
            container: 'body'
        });
    });
}

function openTraceFromUrl(downloadUrl, fileName) {
    const statusDiv = document.getElementById('status');
    statusDiv.style.display = 'none';
    showStatus(`Loading ${fileName}...`, 'success');

    try {
        // Fetch the trace file from the server
        fetch(downloadUrl)
            .then(response => {
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                return response.arrayBuffer();
            })
            .then(traceData => {
                // Open Perfetto UI
                const perfettoWindow = window.open('https://ui.perfetto.dev', '_blank');
                
                if (!perfettoWindow) {
                    showStatus('Please allow pop-ups for this site', 'error');
                    return;
                }

                // Implement PING/PONG handshake protocol
                let pongReceived = false;
                let pingInterval = null;
                
                // Listen for PONG response from Perfetto UI
                const messageHandler = (evt) => {
                    if (evt.origin !== 'https://ui.perfetto.dev') {
                        return;
                    }
                    
                    // Check if we received a PONG response
                    if (evt.data === 'PONG') {
                        pongReceived = true;
                        console.log('Received PONG from Perfetto UI');
                        
                        // Stop sending PING messages
                        if (pingInterval) {
                            clearInterval(pingInterval);
                            pingInterval = null;
                        }
                        
                        // Now send the trace data
                        try {
                            perfettoWindow.postMessage({
                                perfetto: {
                                    buffer: traceData,
                                    title: fileName,
                                    fileName: fileName,
                                    url: downloadUrl
                                }
                            }, 'https://ui.perfetto.dev');
                            
                            showStatus(`✓ Trace loaded successfully: ${fileName}`, 'success');
                            console.log('Trace data sent to Perfetto UI');
                            
                            // Clean up the message listener after sending
                            window.removeEventListener('message', messageHandler);
                        } catch (e) {
                            console.error('Failed to send trace data:', e);
                            showStatus('Error sending trace data. Please try again.', 'error');
                        }
                    }
                };
                
                // Register message listener
                window.addEventListener('message', messageHandler);
                
                // Keep sending PING until we get PONG
                let pingCount = 0;
                const maxPings = 100; // Try for 20 seconds (100 * 200ms)
                
                pingInterval = setInterval(() => {
                    if (pongReceived) {
                        clearInterval(pingInterval);
                        return;
                    }
                    
                    pingCount++;
                    console.log(`Sending PING (attempt ${pingCount}/${maxPings})`);
                    
                    try {
                        perfettoWindow.postMessage('PING', 'https://ui.perfetto.dev');
                    } catch (e) {
                        console.warn('Failed to send PING:', e);
                    }
                    
                    if (pingCount >= maxPings) {
                        clearInterval(pingInterval);
                        window.removeEventListener('message', messageHandler);
                        showStatus('Timeout: Perfetto UI did not respond. Please try again.', 'error');
                        console.error('PONG timeout - Perfetto UI did not respond');
                    }
                }, 200); // Send PING every 200ms
            })
            .catch(error => {
                console.error('Error fetching trace:', error);
                showStatus('Error loading trace from server. Please try downloading manually.', 'error');
            });
    } catch (error) {
        console.error('Error loading trace:', error);
        showStatus('Error loading trace. Please try again.', 'error');
    }
}

document.addEventListener('DOMContentLoaded', () => {
    initializeTooltips();

    document.querySelectorAll('.js-open-trace').forEach(button => {
        button.addEventListener('click', () => {
            openTraceFromUrl(button.dataset.traceUrl, button.dataset.traceName);
        });
    });
});