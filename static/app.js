const chatMessages = document.getElementById('chat-messages');
const chatForm = document.getElementById('chat-form');
const messageInput = document.getElementById('message-input');
const sendBtn = document.getElementById('send-btn');
const fileInput = document.getElementById('file-input');
const uploadStatus = document.getElementById('upload-status');
const docList = document.getElementById('doc-list');

// Load existing documents on page load
async function loadDocuments() {
    try {
        const res = await fetch('/documents');
        const data = await res.json();
        if (data.documents.length > 0) {
            docList.textContent = 'Documents: ' + data.documents.join(', ');
        }
    } catch (e) {
        console.error('Failed to load documents:', e);
    }
}
loadDocuments();

// File upload
fileInput.addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    uploadStatus.textContent = 'Uploading...';
    const formData = new FormData();
    formData.append('file', file);

    try {
        const res = await fetch('/upload', { method: 'POST', body: formData });
        const data = await res.json();
        uploadStatus.textContent = data.message;
        loadDocuments();
    } catch (err) {
        uploadStatus.textContent = 'Upload failed: ' + err.message;
    }
    fileInput.value = '';
});

// Chat
chatForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const message = messageInput.value.trim();
    if (!message) return;

    addMessage(message, 'user');
    messageInput.value = '';
    sendBtn.disabled = true;

    const assistantDiv = addMessage('', 'assistant');

    try {
        const res = await fetch('/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message }),
        });

        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let fullText = '';

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            const chunk = decoder.decode(value, { stream: true });
            const lines = chunk.split('\n');

            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    const data = line.slice(6);
                    if (data === '[DONE]') continue;
                    fullText += data;
                    assistantDiv.textContent = fullText;
                }
            }
        }

        if (!fullText) {
            assistantDiv.textContent = 'No response received.';
        }
    } catch (err) {
        assistantDiv.textContent = 'Error: ' + err.message;
    }

    sendBtn.disabled = false;
    messageInput.focus();
    chatMessages.scrollTop = chatMessages.scrollHeight;
});

function addMessage(text, role) {
    const div = document.createElement('div');
    div.className = `message ${role}`;
    div.textContent = text;
    chatMessages.appendChild(div);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    return div;
}
