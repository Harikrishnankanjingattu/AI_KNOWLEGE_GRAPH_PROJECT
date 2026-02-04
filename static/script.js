document.addEventListener('DOMContentLoaded', () => {
    const chatMessages = document.getElementById('chatMessages');
    const userInput = document.getElementById('userInput');
    const sendBtn = document.getElementById('sendBtn');
    const clearChat = document.getElementById('clearChat');


    userInput.addEventListener('input', function () {
        this.style.height = 'auto';
        this.style.height = (this.scrollHeight) + 'px';
        if (this.scrollHeight > 200) {
            this.style.overflowY = 'scroll';
        } else {
            this.style.overflowY = 'hidden';
        }
    });

    const addMessage = (text, isUser = false) => {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${isUser ? 'user' : 'bot'}`;

        let formattedText = text;
        if (!isUser) {
            formattedText = text
                .replace(/\033\[1;32m/g, '<span style="color: #10b981; font-weight: bold; border-bottom: 1px solid rgba(16, 185, 129, 0.2); margin-bottom: 8px; display: inline-block; width: 100%;">')
                .replace(/\033\[1;33m/g, '<span style="color: #38bdf8; font-weight: bold;">')
                .replace(/\033\[1;31m/g, '<span style="color: #ef4444; font-weight: bold;">')
                .replace(/\033\[1;30m/g, '<span style="color: #94a3b8; font-style: italic;">')
                .replace(/\033\[0m/g, '</span>')
                .replace(/\[\[(.*?)\]\]/g, '<a href="/view_source/$1" target="_blank" style="color: #6366f1; text-decoration: underline;">$1</a>')
                .replace(/\n/g, '<br>');
        }

        messageDiv.innerHTML = `
            <div class="message-inner">
                <div class="message-avatar">${isUser ? 'ME' : 'AI'}</div>
                <div class="message-content">
                    <p>${formattedText}</p>
                </div>
            </div>
        `;

        chatMessages.appendChild(messageDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    };

    const showTyping = () => {
        const typingDiv = document.createElement('div');
        typingDiv.className = 'message bot typing-indicator';
        typingDiv.innerHTML = `
            <div class="message-inner">
                <div class="message-avatar">AI</div>
                <div class="message-content">
                    <div class="typing">
                        <div class="dot"></div>
                        <div class="dot"></div>
                        <div class="dot"></div>
                    </div>
                </div>
            </div>
        `;
        chatMessages.appendChild(typingDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
        return typingDiv;
    };

    const handleSend = async () => {
        const query = userInput.value.trim();
        if (!query) return;

        userInput.value = '';
        userInput.style.height = 'auto';
        addMessage(query, true);

        const typingIndicator = showTyping();

        try {
            const response = await fetch('/query', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query })
            });

            const data = await response.json();
            typingIndicator.remove();
            addMessage(data.answer);
        } catch (error) {
            typingIndicator.remove();
            addMessage("I'm sorry, I encountered an error connecting to the enterprise server.");
            console.error(error);
        }
    };

    sendBtn.addEventListener('click', handleSend);
    userInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSend();
        }
    });

    clearChat.addEventListener('click', () => {
        chatMessages.innerHTML = '';
        addMessage("Chat history cleared. How can I help you?");
    });
});
