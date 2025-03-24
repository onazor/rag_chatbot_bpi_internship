document.addEventListener('DOMContentLoaded', function() {
    // DOM Elements
    const chatMessages = document.getElementById('chat-messages');
    const userInput = document.getElementById('user-input');
    const sendButton = document.getElementById('send-btn');
    const resetButton = document.getElementById('reset-btn');
    const fileInput = document.getElementById('file-input');
    const uploadForm = document.getElementById('upload-form');
    const fileList = document.getElementById('file-list');
    const processingModal = document.getElementById('processing-modal');
    const processingMessage = document.getElementById('processing-message');
    const functionalityItems = document.querySelectorAll('.functionality-item');
    const toggleFilesBtn = document.getElementById('toggle-files-btn');
    const rightSidebar = document.getElementById('right-sidebar');
    
    // Load chat history on page load
    loadChatHistory();
    
    // Toggle files sidebar
    // Check local storage for saved sidebar state
    const isSidebarHidden = localStorage.getItem('filesSidebarHidden') === 'true';

    // Apply initial state
    if (isSidebarHidden) {
        rightSidebar.classList.add('hidden');
        toggleFilesBtn.innerHTML = '<i class="fas fa-file-alt"></i>';
    } else {
        toggleFilesBtn.innerHTML = '<i class="fas fa-times"></i>';
    }

    // Toggle sidebar when button is clicked
    toggleFilesBtn.addEventListener('click', () => {
        rightSidebar.classList.toggle('hidden');
        
        // Update button icon
        if (rightSidebar.classList.contains('hidden')) {
            toggleFilesBtn.innerHTML = '<i class="fas fa-file-alt"></i>';
            localStorage.setItem('filesSidebarHidden', 'true');
        } else {
            toggleFilesBtn.innerHTML = '<i class="fas fa-times"></i>';
            localStorage.setItem('filesSidebarHidden', 'false');
        }
        
        // Trigger resize event to ensure chat container adjusts
        window.dispatchEvent(new Event('resize'));
    });
    
    // Function to handle functionality item clicks
    functionalityItems.forEach(item => {
        item.addEventListener('click', function() {
            const functionType = this.getAttribute('data-function');
            handleFunctionalityClick(functionType);
        });
    });
    
    // Function to handle business functionality selection
    function handleFunctionalityClick(functionType) {
        let promptText = '';
        
        switch(functionType) {
            case 'report-generation':
                showReportGenerationModal();
                return; // Don't continue with setting text in input
                
            case 'risk-assessment':
                promptText = "I need to perform a client risk assessment. Can you help me analyze client financial profiles to evaluate creditworthiness, predict default risks, and identify potential high-value leads?";
                break;
                
            case 'intelligent-query':
                promptText = "I'd like to use the intelligent query handling system. Can you show me how to handle client inquiries regarding loan eligibility, credit applications, and campaign details?";
                break;
                
            case 'industry-classification':
                showIndustryClassificationModal();
                return; // Don't continue with setting text in input
                
            default:
                return;
        }
        
        // Set the input value
        userInput.value = promptText;
        
        // Adjust the height of the textarea
        userInput.style.height = 'auto';
        userInput.style.height = (userInput.scrollHeight) + 'px';
        
        // Focus on the input
        userInput.focus();
    }
    
    // Industry Classification Modal
    function showIndustryClassificationModal() {
        // Create modal if it doesn't exist
        let industryModal = document.getElementById('industry-modal');
        
        if (!industryModal) {
            industryModal = document.createElement('div');
            industryModal.id = 'industry-modal';
            industryModal.className = 'modal';
            
            const modalContent = document.createElement('div');
            modalContent.className = 'modal-content industry-modal-content';
            
            modalContent.innerHTML = `
                <h2>Automated Industry Classification</h2>
                <p>Upload transaction data or use sample data to automatically classify the industry sector.</p>
                
                <div class="industry-options">
                    <div class="industry-option">
                        <h3>Upload Transaction Data</h3>
                        <p>Upload CSV, Excel, or Parquet files containing transaction details.</p>
                        <form id="industry-upload-form">
                            <label for="industry-file-input" class="file-input-label">
                                <i class="fas fa-upload"></i> Upload Transaction Data
                            </label>
                            <input type="file" id="industry-file-input" name="file" style="display: none;" accept=".csv,.xlsx,.parquet">
                        </form>
                    </div>
                    
                    <div class="industry-option">
                        <h3>Use Sample Data</h3>
                        <p>Try with pre-loaded sample transaction data for retail, restaurant, or tech sector.</p>
                        <button id="sample-data-btn" class="industry-btn">Use Sample Data</button>
                        <p class="sample-note">Uses a simple retail transaction dataset with inventory purchases, sales, and operating expenses.</p>
                    </div>
                </div>
                
                <div class="industry-footer">
                    <button id="close-industry-modal-btn">Cancel</button>
                </div>
            `;
            
            industryModal.appendChild(modalContent);
            document.body.appendChild(industryModal);
            
            // Set up event listeners
            document.getElementById('close-industry-modal-btn').addEventListener('click', () => {
                industryModal.classList.remove('show');
            });
            
            document.getElementById('industry-file-input').addEventListener('change', handleIndustryFileUpload);
            
            document.getElementById('sample-data-btn').addEventListener('click', () => {
                processIndustryClassification(null, true);
                industryModal.classList.remove('show');
            });
        }
        
        // Show the modal
        industryModal.classList.add('show');
    }
    
    // Function to show Report Generation Modal
    function showReportGenerationModal() {
        console.log("Opening report generation modal");
        
        // Create the modal if it doesn't exist
        let reportModal = document.getElementById('report-modal');
        
        if (!reportModal) {
            reportModal = document.createElement('div');
            reportModal.id = 'report-modal';
            reportModal.className = 'modal';
            
            const modalContent = document.createElement('div');
            modalContent.className = 'modal-content industry-modal-content';
            
            modalContent.innerHTML = `
                <h2>Generate Financial Report</h2>
                <p>Upload a financial document to generate a detailed analysis report.</p>
                
                <div class="industry-options">
                    <div class="industry-option">
                        <h3>Upload Financial Document</h3>
                        <p>Upload a PDF, Excel, or CSV file containing financial data for analysis.</p>
                        <form id="report-form">
                            <div class="form-group">
                                <label for="report-type" style="display: block; margin-bottom: 8px; font-weight: 500;">Report Type:</label>
                                <select name="reportType" id="report-type" style="width: 100%; padding: 8px; margin-bottom: 15px; border: 1px solid #ddd; border-radius: 4px;">
                                    <option value="comprehensive">Comprehensive Analysis</option>
                                    <option value="loan-portfolio">Loan Portfolio Analysis</option>
                                    <option value="financial-metrics">Financial Metrics Analysis</option>
                                    <option value="campaign-performance">Campaign Performance Analysis</option>
                                    <option value="text-summary">Simple Text Summarization</option>
                                </select>
                            </div>
                            
                            <label for="report-file" class="file-input-label" style="display: block; margin-bottom: 10px; text-align: center;">
                                <i class="fas fa-upload"></i> Upload Document
                            </label>
                            <input type="file" name="file" id="report-file" style="display: none;" accept=".pdf,.xlsx,.xls,.csv">
                            <p class="sample-note">Supports PDF, Excel, and CSV files.</p>
                            
                            <button type="submit" class="industry-btn">Generate Report</button>
                        </form>
                    </div>
                </div>
                
                <div id="report-progress" style="display: none; text-align: center; margin-top: 20px;">
                    <p style="margin-bottom: 10px;">Generating report... This may take up to 30 seconds.</p>
                    <div class="spinner" style="border: 4px solid rgba(183, 12, 26, 0.3); border-radius: 50%; border-top: 4px solid #b70c1a; width: 30px; height: 30px; margin: 0 auto; animation: spin 1s linear infinite;"></div>
                </div>
                
                <div class="industry-footer">
                    <button id="close-report-modal-btn">Cancel</button>
                </div>
            `;
            
            reportModal.appendChild(modalContent);
            document.body.appendChild(reportModal);
            
            // Set up event listeners
            document.getElementById('close-report-modal-btn').addEventListener('click', () => {
                reportModal.classList.remove('show');
            });
            
            const reportFileInput = document.getElementById('report-file');
            reportFileInput.addEventListener('change', function() {
                // Show the file name next to the upload button if a file is selected
                if (this.files.length > 0) {
                    const fileNameElem = document.createElement('span');
                    fileNameElem.textContent = this.files[0].name;
                    fileNameElem.style.marginLeft = '10px';
                    fileNameElem.style.fontStyle = 'italic';
                    
                    // Remove any existing filename
                    const existingFileName = reportFileInput.nextElementSibling.nextElementSibling;
                    if (existingFileName && existingFileName.tagName === 'SPAN') {
                        existingFileName.remove();
                    }
                    
                    reportFileInput.parentNode.insertBefore(fileNameElem, reportFileInput.nextElementSibling.nextElementSibling);
                }
            });
            
            // Form submission
            const form = document.getElementById('report-form');
            const progressDiv = document.getElementById('report-progress');
            
            form.addEventListener('submit', async (e) => {
                e.preventDefault();
                
                const fileInput = document.getElementById('report-file');
                if (!fileInput.files.length) {
                    alert('Please select a file to upload');
                    return;
                }
                
                // Show progress
                form.style.display = 'none';
                progressDiv.style.display = 'block';
                
                const reportType = document.getElementById('report-type').value;
                
                // Create form data
                const formData = new FormData();
                formData.append('file', fileInput.files[0]);
                
                // Add settings as JSON
                const settings = {
                    reportType: reportType,
                    includeExecutiveSummary: true,
                    includeKeyMetrics: true, 
                    includeRecommendations: true
                };
                formData.append('settings', JSON.stringify(settings));
                
                try {
                    showProcessingModal('Generating your report, please wait...');
                    
                    const response = await fetch('/generate_report', {
                        method: 'POST',
                        body: formData
                    });
                    
                    // Check if response is JSON
                    const contentType = response.headers.get('content-type');
                    if (!contentType || !contentType.includes('application/json')) {
                        throw new Error(`Unexpected response format: ${contentType}`);
                    }
                    
                    const data = await response.json();
                    
                    hideProcessingModal();
                    
                    if (data.error) {
                        throw new Error(data.error);
                    }
                    
                    // Close the modal
                    reportModal.classList.remove('show');
                    
                    // Add message to chat
                    addMessage(data.details, false);
                    
                    // Add download link if available
                    if (data.reportUrl) {
                        // Create a cleaner markdown message without extra indentation
                        const downloadMsg = `## Download Report

Your report is ready. [Click here to download the PDF report](${data.reportUrl})`;
                        
                        addMessage(downloadMsg, false);
                        
                        // Add an event listener to handle download errors
                        setTimeout(() => {
                            const downloadLinks = document.querySelectorAll('.message-content a[href^="/download_report/"]');
                            downloadLinks.forEach(link => {
                                // Keep the original click event but add error handling
                                link.addEventListener('click', function(e) {
                                    // Don't prevent default - let the browser handle the navigation
                                    // But add a catch in case it returns an error JSON
                                    console.log('Download link clicked:', link.href);
                                });
                            });
                        }, 500);
                    }
                    
                } catch (error) {
                    console.error("Report generation error:", error);
                    
                    hideProcessingModal();
                    
                    // Close the modal
                    reportModal.classList.remove('show');
                    
                    // Show error in chat
                    addMessage(`## An error occurred while generating the report: ${error.message}`, false);
                }
            });
        }
        
        // Show the modal
        reportModal.classList.add('show');
        
        // Reset the modal state
        const form = document.getElementById('report-form');
        const progressDiv = document.getElementById('report-progress');
        
        // Reset form display
        if (form) form.style.display = 'block';
        
        // Hide progress indicator
        if (progressDiv) progressDiv.style.display = 'none';
        
        // Ensure the file input is cleared when opening the modal
        const fileInput = document.getElementById('report-file');
        if (fileInput) {
            fileInput.value = '';
            
            // Remove any displayed filename if present
            const fileNameSpan = fileInput.parentNode.querySelector('span');
            if (fileNameSpan) {
                fileNameSpan.remove();
            }
        }
    }
    
    // Function to handle industry classification file upload
    async function handleIndustryFileUpload(e) {
        const file = e.target.files[0];
        if (!file) return;
        
        document.getElementById('industry-modal').classList.remove('show');
        processIndustryClassification(file);
    }
    
    // Function to process industry classification
    async function processIndustryClassification(file, useSampleData = false) {
        try {
            showProcessingModal('Processing industry classification...');
            
            let response;
            
            if (file) {
                // Upload actual file
                const formData = new FormData();
                formData.append('file', file);
                
                response = await fetch('/classify_industry', {
                    method: 'POST',
                    body: formData
                });
            } else if (useSampleData) {
                // Use sample data
                response = await fetch('/classify_industry', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ useSampleData: true })
                });
            }
            
            const data = await response.json();
            
            hideProcessingModal();
            
            if (data.error) {
                console.error('Classification error:', data.error);
                addMessage(`Error processing industry classification: ${data.error}`, false);
            } else if (data.success) {
                if (file) {
                    // Refresh file list since we uploaded a new file
                    const filesResponse = await fetch('/files');
                    const filesData = await filesResponse.json();
                    updateFileList(filesData.files);
                }
                
                // Don't rely on loadChatHistory which may clear messages
                // Instead display the classification directly in the chat
                if (data.classification) {
                    addMessage(`## Industry Classification Results\n\n${data.classification}`, false);
                } else {
                    addMessage(`Successfully processed ${file ? file.name : 'sample data'} for industry classification. The detailed results should appear shortly.`, false);
                }
            }
        } catch (error) {
            console.error('Error during industry classification:', error);
            hideProcessingModal();
            addMessage(`An error occurred while processing the industry classification: ${error.message}`, false);
        }
    }
    
    // Function to load chat history from server
    async function loadChatHistory() {
        try {
            showProcessingModal('Loading chat history...');
            
            const response = await fetch('/chat_history');
            const data = await response.json();
            
            if (data.chat_history && data.chat_history.length > 0) {
                // Clear the default welcome message
                chatMessages.innerHTML = '';
                
                // Add all messages from history
                data.chat_history.forEach(msg => {
                    if (msg.role !== 'system') {
                        addMessage(msg.content, msg.role === 'user');
                    }
                });
            }
        } catch (error) {
            console.error('Error loading chat history:', error);
        } finally {
            hideProcessingModal();
        }
    }
    
    // Function to add a message to the chat
    function addMessage(content, isUser = false) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${isUser ? 'user' : 'assistant'}`;
        
        const messageContent = document.createElement('div');
        messageContent.className = 'message-content';
        
        if (isUser) {
            // For user messages, just use plain text
            const messageParagraph = document.createElement('p');
            messageParagraph.textContent = content;
            messageContent.appendChild(messageParagraph);
        } else {
            // For bot messages, render as Markdown
            // Set up marked.js options
            marked.setOptions({
                breaks: true,          // Add line breaks with a single newline
                gfm: true,             // Enable GitHub Flavored Markdown
                headerIds: false,      // Disable automatic IDs for headers
                mangle: false,         // Disable escaping HTML
                sanitize: false        // Let DOMPurify handle sanitization
            });
            
            // Convert markdown to HTML and sanitize
            const rawHtml = marked.parse(content);
            const sanitizedHtml = DOMPurify.sanitize(rawHtml, {
                ADD_ATTR: ['target'] // Allow target attribute for links
            });
            
            // Insert the HTML
            messageContent.innerHTML = sanitizedHtml;
            
            // Make all links open in new tab
            const links = messageContent.querySelectorAll('a');
            links.forEach(link => {
                link.setAttribute('target', '_blank');
                link.setAttribute('rel', 'noopener noreferrer');
            });
        }
        
        messageDiv.appendChild(messageContent);
        chatMessages.appendChild(messageDiv);
        
        // Scroll to bottom of chat
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
    
    // Function to show typing indicator
    function showTypingIndicator() {
        const typingDiv = document.createElement('div');
        typingDiv.className = 'message assistant typing';
        typingDiv.id = 'typing-indicator';
        
        for (let i = 0; i < 3; i++) {
            const dot = document.createElement('span');
            typingDiv.appendChild(dot);
        }
        
        chatMessages.appendChild(typingDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
    
    // Function to remove typing indicator
    function removeTypingIndicator() {
        const typingIndicator = document.getElementById('typing-indicator');
        if (typingIndicator) {
            typingIndicator.remove();
        }
    }
    
    // Function to show processing modal
    function showProcessingModal(message = 'Processing your request...') {
        processingMessage.textContent = message;
        processingModal.classList.add('show');
        
        // Add a timeout indicator that changes the message after 15 seconds
        setTimeout(() => {
            if (processingModal.classList.contains('show')) {
                processingMessage.textContent = message + ' (This might take a moment...)';
            }
        }, 15000);
    }
    
    // Function to hide processing modal
    function hideProcessingModal() {
        processingModal.classList.remove('show');
    }
    
    // Function to update file list
    function updateFileList(files) {
        const ul = fileList.querySelector('ul');
        ul.innerHTML = '';
        
        files.forEach(file => {
            const li = document.createElement('li');
            
            const span = document.createElement('span');
            span.className = 'file-name';
            span.textContent = file;
            
            const deleteButton = document.createElement('button');
            deleteButton.className = 'delete-file-btn';
            deleteButton.setAttribute('data-filename', file);
            deleteButton.innerHTML = '<i class="fas fa-trash"></i>';
            
            li.appendChild(span);
            li.appendChild(deleteButton);
            ul.appendChild(li);
        });
        
        // Add event listeners to delete buttons
        document.querySelectorAll('.delete-file-btn').forEach(button => {
            button.addEventListener('click', handleDeleteFile);
        });
    }
    
    // Function to handle file deletion
    async function handleDeleteFile(e) {
        const filename = e.currentTarget.getAttribute('data-filename');
        if (!confirm(`Are you sure you want to delete ${filename}?`)) return;
        
        try {
            showProcessingModal(`Deleting ${filename}...`);
            
            const response = await fetch(`/delete_file/${filename}`, {
                method: 'POST'
            });
            
            const data = await response.json();
            
            if (data.success) {
                updateFileList(data.files);
                addMessage(`File '${filename}' has been deleted.`, false);
            } else {
                console.error('Error:', data.error);
                addMessage(`Error deleting file: ${data.error}`, false);
            }
        } catch (error) {
            console.error('Error:', error);
            addMessage('An error occurred while deleting the file.', false);
        } finally {
            hideProcessingModal();
        }
    }
    
    // Function to send message to API
    async function sendMessage(message) {
        if (!message.trim()) return;
        
        // Add user message to chat
        addMessage(message, true);
        
        // Clear input field
        userInput.value = '';
        userInput.style.height = 'auto';
        
        // Show typing indicator
        showTypingIndicator();
        
        // Set up request timeout
        const timeoutDuration = 60000; // 60 seconds
        let timeoutId;
        
        try {
            // Create an AbortController to cancel the fetch if it takes too long
            const controller = new AbortController();
            const signal = controller.signal;
            
            // Set up timeout to cancel the fetch after timeoutDuration
            timeoutId = setTimeout(() => {
                controller.abort();
                throw new Error('Request timed out');
            }, timeoutDuration);
            
            // Send message to API
            const response = await fetch('/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ message }),
                signal: signal
            });
            
            // Clear timeout since request completed
            clearTimeout(timeoutId);
            
            const data = await response.json();
            
            // Remove typing indicator
            removeTypingIndicator();
            
            if (data.error) {
                console.error('Error:', data.error);
                addMessage("I'm sorry, I encountered an error processing your request. Please try again with a more specific question.", false);
            } else {
                // Add assistant's response to chat
                addMessage(data.message);
            }
        } catch (error) {
            // Clear timeout if it's still active
            if (timeoutId) clearTimeout(timeoutId);
            
            console.error('Error:', error);
            // Remove typing indicator
            removeTypingIndicator();
            
            // Show appropriate error message
            if (error.name === 'AbortError' || error.message === 'Request timed out') {
                addMessage("I'm sorry, but your request took too long to process. Please try asking a simpler question or breaking it down into smaller parts.", false);
            } else {
                addMessage("I'm sorry, an error occurred while processing your request. Please try again.", false);
            }
        }
    }
    
    // Function to handle file upload
    async function handleFileUpload(e) {
        const file = fileInput.files[0];
        if (!file) return;
        
        const formData = new FormData();
        formData.append('file', file);
        
        showProcessingModal(`Uploading and processing ${file.name}...`);
        
        try {
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });
            
            const data = await response.json();
            
            if (data.success) {
                // Update file list
                updateFileList(data.files);
                
                // Add success message to chat
                addMessage(`File '${file.name}' has been uploaded and processed successfully.`, false);
            } else {
                console.error('Error:', data.error);
                addMessage(`Error uploading file: ${data.error}`, false);
            }
        } catch (error) {
            console.error('Error:', error);
            addMessage('An error occurred while uploading the file.', false);
        } finally {
            // Reset file input
            uploadForm.reset();
            hideProcessingModal();
        }
    }
    
    // Event listener for file input change
    fileInput.addEventListener('change', handleFileUpload);
    
    // Event listener for send button
    sendButton.addEventListener('click', function() {
        sendMessage(userInput.value);
    });
    
    // Event listener for enter key
    userInput.addEventListener('keydown', function(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage(userInput.value);
        }
    });
    
    // Auto-resize textarea as user types
    userInput.addEventListener('input', function() {
        this.style.height = 'auto';
        this.style.height = (this.scrollHeight) + 'px';
    });
    
    // Event listener for reset button
    resetButton.addEventListener('click', async function() {
        if (!confirm("Are you sure you want to start a new chat? This will clear your conversation history.")) return;
        
        try {
            showProcessingModal('Starting a new chat...');
            
            const response = await fetch('/reset', {
                method: 'POST'
            });
            
            const data = await response.json();
            
            if (data.success) {
                // Clear chat messages
                chatMessages.innerHTML = '';
                // Add welcome message with Markdown formatting
                addMessage('## Hi, I am Bria! Your assistant for BPI Business Banking!\n\nI can help you with:\n\n- **Automated Reports** for loan portfolios and campaigns\n- **Risk Assessment** and lead identification\n- **Query Handling** for client inquiries\n- **Document Analysis** from your uploaded files\n- **Industry Classification** for targeted credit and loan campaigns\n\nSelect a function from the left panel or ask me a question directly.');
                
                // Force un-hide the files sidebar
                rightSidebar.classList.remove('hidden');
                toggleFilesBtn.innerHTML = '<i class="fas fa-times"></i>';
                localStorage.setItem('filesSidebarHidden', 'false');
            }
        } catch (error) {
            console.error('Error:', error);
            addMessage('An error occurred while resetting the chat.');
        } finally {
            hideProcessingModal();
        }
    });
    
    // Add event listeners to existing delete buttons
    document.querySelectorAll('.delete-file-btn').forEach(button => {
        button.addEventListener('click', handleDeleteFile);
    });

    // Set up the sidebar menu handlers
    const reportGenBtn = document.getElementById('report-gen-btn');
    if (reportGenBtn) {
        reportGenBtn.addEventListener('click', function() {
            showReportGenerationModal();
        });
    } else {
        console.error("Report generation button not found in the DOM");
    }
    
    // Add click handler to the whole sidebar to capture dynamically added buttons
    const sidebar = document.querySelector('.sidebar');
    if (sidebar) {
        sidebar.addEventListener('click', function(e) {
            // Check if the clicked element has specific IDs or classes
            if (e.target.id === 'report-gen-btn' || e.target.closest('#report-gen-btn')) {
                showReportGenerationModal();
            } else if (e.target.id === 'industry-class-btn' || e.target.closest('#industry-class-btn')) {
                showIndustryClassificationModal();
            }
        });
    }
}); 