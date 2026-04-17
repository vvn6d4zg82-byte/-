document.getElementById('start').onclick = function() {
  chrome.tabs.query({active: true, currentWindow: true}, function(tabs) {
    if (!tabs || !tabs[0]) {
      document.getElementById('msg').textContent = '无标签页';
      return;
    }
    
    // 先发送ping测试连接
    chrome.tabs.sendMessage(tabs[0].id, {action: 'ping'}, function(response) {
      if (chrome.runtime.lastError) {
        console.log('ping失败:', chrome.runtime.lastError.message);
        document.getElementById('msg').textContent = '请刷新页面';
        return;
      }
      
      // 连接成功，发送start
      chrome.tabs.sendMessage(tabs[0].id, {action: 'start'}, function() {
        document.getElementById('start').disabled = true;
        document.getElementById('stop').disabled = false;
        document.getElementById('status').textContent = '状态: 运行中';
        document.getElementById('status').className = 'running';
        document.getElementById('msg').textContent = '';
      });
    });
  });
};

document.getElementById('stop').onclick = function() {
  chrome.tabs.query({active: true, currentWindow: true}, function(tabs) {
    if (tabs && tabs[0]) {
      chrome.tabs.sendMessage(tabs[0].id, {action: 'stop'}, function() {
        document.getElementById('stop').disabled = true;
        document.getElementById('start').disabled = false;
        document.getElementById('status').textContent = '状态: 已停止';
        document.getElementById('status').className = 'stopped';
      });
    }
  });
};