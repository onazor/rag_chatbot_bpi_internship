// Load chat history and render messages appropriately
function loadChatHistory() {
  fetch("/chat_history")
    .then(response => response.json())
    .then(data => {
      let chatBox = document.getElementById("chat-box");
      data.chat_history.forEach(msg => {
        let messageDiv = document.createElement("div");
        if (msg.role === "bot") {
          // Render bot messages as markdown HTML
          messageDiv.className = "bot-message";
          messageDiv.innerHTML = marked.parse(msg.content);
        } else {
          // Render user messages as plain text
          messageDiv.className = "user-message";
          messageDiv.innerText = msg.content;
        }
        chatBox.appendChild(messageDiv);
      });
      chatBox.scrollTop = chatBox.scrollHeight;
    })
    .catch(error => {
      console.error("Error loading chat history:", error);
    });
}

// Listen for Enter key on the question textarea
document.addEventListener("DOMContentLoaded", function() {
  loadChatHistory();

  document.getElementById("question").addEventListener("keydown", function(e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      askQuestion();
    }
  });
});

// Trigger file upload
function triggerFileUpload() {
  document.getElementById("uploadFile").click();
}

// When a file is selected, show a toast message with the file name
function handleFileSelect() {
  const fileInput = document.getElementById("uploadFile");
  const chatBox = document.getElementById("chat-box");

  if (fileInput.files.length > 0) {
      const file = fileInput.files[0];
      const fileName = file.name;
      
      // Create the container for the file upload message
      const fileUploadDiv = document.createElement("div");
      fileUploadDiv.className = "file-upload-message user-message";
      // "user-message" positions it on the right side

      // Optional file icon
      const fileIcon = document.createElement("div");
      fileIcon.className = "file-icon";

      // File details container
      const fileDetails = document.createElement("div");
      fileDetails.className = "file-details";

      // Filename text
      const fileNameP = document.createElement("p");
      fileNameP.className = "file-name";
      fileNameP.innerText = fileName;

      // Status text (start blank instead of "Preparing to upload...")
      const fileStatusP = document.createElement("p");
      fileStatusP.className = "file-status";
      fileStatusP.innerText = ""; 

      // Append elements
      fileDetails.appendChild(fileNameP);
      fileDetails.appendChild(fileStatusP);
      fileUploadDiv.appendChild(fileIcon);
      fileUploadDiv.appendChild(fileDetails);

      chatBox.appendChild(fileUploadDiv);
      chatBox.scrollTop = chatBox.scrollHeight;
      showUploadStatus(`File selected: ${fileName}`);
  }
}

// Toast for file upload status
function showUploadStatus(message) {
  const container = document.getElementById('upload-status-container');
  const msg = document.getElementById('upload-status-message');
  msg.innerText = message;

  container.classList.add('show');
  setTimeout(() => {
    container.classList.remove('show');
  }, 4000);
}

function typeMessage(element, fullText, delay = 25) {
  // Split the full text into words
  const words = fullText.split(" ");
  let currentText = "";
  let i = 0;
  
  function addWord() {
    if (i < words.length) {
      // Append the next word (with a space if not the first word)
      currentText += (i > 0 ? " " : "") + words[i];
      // Use marked.parse to process markdown (with breaks enabled)
      element.innerHTML = marked.parse(currentText, { breaks: false });
      i++;
      setTimeout(addWord, delay);
    }
  }
  addWord();
}

// Main function for sending question (and file if provided)
function askQuestion() {
  const questionInput = document.getElementById("question");
  const chatBox = document.getElementById("chat-box");
  const fileInput = document.getElementById("uploadFile");

  const question = questionInput.value.trim();
  if (question === "" && fileInput.files.length === 0) return;

  if (question !== "") {
      const userMessage = document.createElement("div");
      userMessage.className = "user-message";
      userMessage.innerText = question;
      chatBox.appendChild(userMessage);
  }
  questionInput.value = "";

  if (fileInput.files.length > 0) {
      showUploadStatus("Uploading file...");
  }

  // Create a bot message element with typing animation
  const botMessage = document.createElement("div");
  botMessage.className = "bot-message typing";
  botMessage.innerText = "Typing...";
  chatBox.appendChild(botMessage);
  chatBox.scrollTop = chatBox.scrollHeight;

  const formData = new FormData();
  formData.append("question", question);
  if (fileInput.files.length > 0) {
      formData.append("file", fileInput.files[0]);
  }

  fetch("/ask", {
      method: "POST",
      body: formData
  })
  .then(response => response.json())
  .then(data => {
      botMessage.classList.remove("typing");
      let finalAnswer = data.answer;
      finalAnswer = finalAnswer.replace(/\n{3,}/g, "\n\n");
      botMessage.innerHTML = marked.parse(finalAnswer);
      typeMessage(botMessage, data.answer, 150);
      if (fileInput.files.length > 0) {
          showUploadStatus("File upload complete!");
          fileInput.value = "";
      }
  })
  .catch(error => {
      botMessage.classList.remove("typing");
      botMessage.innerText = "Error: Unable to get response.";
      if (fileInput.files.length > 0) {
          showUploadStatus("File upload failed!");
      }
      console.error("Upload error:", error);
  });
}

// reset the session
function resetSession() {
  fetch("/reset", { method: "POST" })
    .then(response => response.json())
    .then(data => {
      if (data.success) {
        const chatBox = document.getElementById("chat-box");
        chatBox.innerHTML = '<div class="bot-message">Hi, I am Bria! Feel free to ask me questions related to business!</div>';
        document.getElementById("question").value = "";
        document.getElementById("uploadFile").value = "";
      } else {
        console.error("Error resetting session:", data.error);
      }
    })
    .catch(error => {
      console.error("Error resetting session:", error);
    });
}