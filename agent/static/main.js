document.addEventListener('DOMContentLoaded', function () {
    // 初始化系统状态
    checkSystemStatus();

    // 加载历史记录
    loadHistory();

    // 设置输入框自动聚焦
    document.getElementById('question').focus();

    // 添加快捷键支持
    setupShortcuts();

    // 绑定提问按钮事件
    document.getElementById('ask-button').addEventListener('click', askQuestion);

    // 绑定复制按钮事件
    document.getElementById('copy-button').addEventListener('click', copyResponse);

    // 绑定清空历史按钮事件
    document.getElementById('clear-history').addEventListener('click', clearHistory);

    // 绑定重新生成按钮事件
    document.getElementById('regenerate-button').addEventListener('click', regenerateResponse);
    setInterval(updateCurrentTime, 1000);
    updateCurrentTime();
});

function updateCurrentTime() {
    const now = new Date();
    document.getElementById('current-time').textContent =
        `系统时间：${now.toLocaleString()}`;
}

// 检查系统状态
function checkSystemStatus() {
    fetch('/health')
        .then(response => response.json())
        .then(data => {
            // 更新服务状态
            document.getElementById('service-status').textContent =
                data.status === 'healthy' ? '运行正常' : '服务异常';

            // 更新模型状态
            document.getElementById('model-status').textContent =
                data.model.model_loaded ? `已加载 (${data.model.model_device})` : '未加载';

            // 更新知识库状态
            document.getElementById('kb-status').textContent =
                data.vector_db.vector_db_loaded ?
                    `已加载 (${data.vector_db.vector_count} 条)` : '未加载';

            // 更新响应时间（从健康检查中获取）
            document.getElementById('response-time-status').textContent =
                `${data.response_time} ms`;
        })
        .catch(error => {
            console.error('获取系统状态失败:', error);
            document.getElementById('service-status').textContent = '获取状态失败';
        });
}

// 提问函数
async function askQuestion() {
    // 获取用户输入的问题
    const question = document.getElementById('question').value;
    const maxLength = document.getElementById('max-length').value;
    const topK = document.getElementById('top-k').value;

    // 验证输入
    if (!question.trim()) {
        alert('请输入问题');
        return;
    }

    // 显示加载动画
    const loader = document.getElementById('loader');
    loader.style.display = 'block';

    // 发送请求到后端
    try {
        const response = await fetch('/ask', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                query: question,  // 确保发送实际的问题内容
                max_length: parseInt(maxLength),
                top_k: parseInt(topK)
            })
        });
        if (!response.ok) throw new Error(`请求失败: ${response.status}`);
        const data = await response.json();

        // 更新UI显示响应
        document.getElementById('query-text').textContent = data.query;
        document.getElementById('answer-text').textContent = data.response; // 使用实际响应
        document.getElementById('response-time').textContent =
            `耗时：${data.response_time} ms`; // 使用毫秒显示

        // 显示响应区域
        document.getElementById('response').style.display = 'block';
    } catch (error) {
        console.error('请求出错:', error);
        alert('请求出错: ' + error.message);
    } finally {
        loader.style.display = 'none';
    }
}


// 重新生成回答
function regenerateResponse() {
    const question = document.getElementById('question').value;
    if (question.trim()) {
        askQuestion();
    } else {
        alert('请先输入问题');
    }
}

// 复制回答到剪贴板
function copyResponse() {
    const answerText = document.getElementById('answer-text').textContent;
    navigator.clipboard.writeText(answerText)
        .then(() => {
            // 显示复制成功的提示
            const copyBtn = document.getElementById('copy-button');
            const originalText = copyBtn.textContent;
            copyBtn.innerHTML = '<i class="fas fa-check"></i> 已复制';

            // 2秒后恢复原始文本
            setTimeout(() => {
                copyBtn.innerHTML = '<i class="fas fa-copy"></i> 复制回答';
            }, 2000);
        })
        .catch(err => {
            console.error('复制失败:', err);
            alert('复制失败，请手动复制');
        });
}

// 设置快捷键支持
function setupShortcuts() {
    // 添加Ctrl+Enter提交问题的快捷键
    document.getElementById('question').addEventListener('keydown', function (e) {
        if (e.ctrlKey && e.key === 'Enter') {
            askQuestion();
        }
    });

    // 添加历史记录快捷键
    document.addEventListener('keydown', function (e) {
        if (e.key === 'Escape') {
            document.getElementById('question').focus();
        }
    });
}

// 添加到历史记录
function addToHistory(question, answer) {
    // 获取现有历史记录或初始化空数组
    const history = JSON.parse(localStorage.getItem('ai_assistant_history')) || [];

    // 创建新的历史记录项
    const historyItem = {
        id: Date.now(),
        question: question,
        answer: answer,
        timestamp: new Date().toISOString()
    };

    // 添加到历史记录数组开头
    history.unshift(historyItem);

    // 限制历史记录数量（最多20条）
    if (history.length > 20) {
        history.pop();
    }

    // 保存到本地存储
    localStorage.setItem('ai_assistant_history', JSON.stringify(history));

    // 更新UI显示
    loadHistory();
}

// 加载历史记录
function loadHistory() {
    const historyContainer = document.getElementById('history-container');
    const history = JSON.parse(localStorage.getItem('ai_assistant_history')) || [];

    // 清空容器
    historyContainer.innerHTML = '';

    if (history.length === 0) {
        historyContainer.innerHTML = '<p>暂无历史记录</p>';
        return;
    }

    // 添加历史记录项
    history.forEach(item => {
        const historyItem = document.createElement('div');
        historyItem.className = 'history-item';
        historyItem.dataset.id = item.id;

        // 创建问题摘要（最多50个字符）
        const questionExcerpt = item.question.length > 50 ?
            item.question.substring(0, 47) + '...' : item.question;

        // 格式化日期
        const date = new Date(item.timestamp);
        const formattedDate = `${date.getFullYear()}-${(date.getMonth() + 1).toString().padStart(2, '0')}-${date.getDate().toString().padStart(2, '0')} ${date.getHours().toString().padStart(2, '0')}:${date.getMinutes().toString().padStart(2, '0')}`;

        historyItem.innerHTML = `
            <h5><i class="fas fa-question-circle"></i> ${questionExcerpt}</h5>
            <div class="history-date">${formattedDate}</div>
        `;

        // 添加点击事件
        historyItem.addEventListener('click', () => {
            loadHistoryItem(item.id);
        });

        historyContainer.appendChild(historyItem);
    });

    // 显示历史记录区域
    document.getElementById('history').style.display = 'block';
}

// 加载历史记录项
function loadHistoryItem(id) {
    const history = JSON.parse(localStorage.getItem('ai_assistant_history')) || [];
    const item = history.find(i => i.id == id);

    if (!item) return;

    // 更新问题输入框
    document.getElementById('question').value = item.question;

    // 更新回答显示
    document.getElementById('query-text').textContent = item.question;
    document.getElementById('answer-text').innerHTML = formatAnswer(item.answer);

    // 清空参考资料
    document.getElementById('sources-container').innerHTML = '<p>历史记录中未保存参考资料</p>';

    // 隐藏加载器，显示响应区域
    document.getElementById('loader').style.display = 'none';
    document.getElementById('response').style.display = 'block';

    // 更新时间显示
    const date = new Date(item.timestamp);
    const formattedDate = `${date.getHours().toString().padStart(2, '0')}:${date.getMinutes().toString().padStart(2, '0')}`;
    document.getElementById('response-time-meta').textContent = `保存于: ${formattedDate}`;
    document.getElementById('response-time').textContent = '--';
}

// 清空历史记录
function clearHistory() {
    if (confirm('确定要清空所有历史记录吗？此操作不可撤销。')) {
        localStorage.removeItem('ai_assistant_history');
        document.getElementById('history-container').innerHTML = '<p>暂无历史记录</p>';
    }
}

// 格式化答案文本
function formatAnswer(text) {
    // 替换换行符为HTML换行
    let formatted = text.replace(/\n/g, '<br>');

    // 识别标题并加粗
    const headings = ['一、', '二、', '三、', '四、', '五、', '六、', '七、', '八、', '九、', '十、'];
    headings.forEach(heading => {
        const regex = new RegExp(`${heading}(.+?)<br>`, 'g');
        formatted = formatted.replace(regex, `<strong>${heading}$1</strong><br>`);
    });

    // 识别列表项
    formatted = formatted.replace(/([1-9]\.\s)/g, '<span style="font-weight: bold; margin-right: 8px;">$1</span>');

    // 识别代码块
    formatted = formatted.replace(/```([\s\S]*?)```/g, '<pre><code>$1</code></pre>');

    return formatted;
}