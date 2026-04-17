chrome.action.onClicked.addListener(async (tab) => {
  if (!tab.url.includes('zhihuishu')) {
    alert('请在智慧树页面使用');
    return;
  }
  
  try {
    await chrome.scripting.executeScript({
      target: { tabId: tab.id },
      files: ['content.js']
    });
    console.log('脚本已注入');
  } catch (e) {
    console.log('注入失败:', e);
  }
});

chrome.runtime.onMessage.addListener((req, sender, sendResponse) => {
  console.log('background收到:', req.action);
  sendResponse({status: 'ok'});
  return true;
});