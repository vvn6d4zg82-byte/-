(function() {
  'use strict';
  console.log('[智慧树] 刷课助手 v2.3 已加载');
  
  let running = false;
  let timer = null;

  function log(msg) { console.log('[智慧树] ' + msg); }

  function start() {
    if (running) return;
    running = true;
    log('开始运行');
    doTask();
    timer = setInterval(doTask, 2000);
  }

  function stop() {
    running = false;
    if (timer) { clearInterval(timer); timer = null; }
    log('已停止');
  }

  function doTask() {
    if (!running) return;
    
    let video = document.querySelector('video');
    if (!video) { 
      log('未找到视频');
      return; 
    }

    // 静音
    video.muted = true;
    video.volume = 0;
    
    // 尝试多种方式设置倍速
    trySetSpeed(video);
    
    // 自动播放
    if (video.paused) {
      video.play().catch(function(){});
    }

    let pct = video.duration > 0 ? (video.currentTime / video.duration * 100).toFixed(0) : 0;
    log('进度: ' + pct + '%');

    // 进度达90%或结束，切换下一节
    if (video.ended || (video.duration > 0 && video.currentTime / video.duration >= 0.9)) {
      log('视频完成，切换下一节');
      playNext();
    }

    // 处理弹窗
    handlePopup();
  }

  function trySetSpeed(video) {
    // 方式1: 直接设置 playbackRate
    video.playbackRate = 1.5;
    
    // 方式2: 尝试点击倍速按钮
    let speedBtn = document.querySelector('.speedTab15, .speedTab, [class*="speed"], .speedBox');
    if (speedBtn) {
      speedBtn.click();
      log('点击了倍速按钮');
    }
    
    // 方式3: 查找倍速选项
    let speedOptions = document.querySelectorAll('.speedTab15, [class*="15"]');
    if (speedOptions.length > 0) {
      speedOptions[0].click();
    }
  }

  function playNext() {
    // 尝试点击下一节按钮
    let btn = document.querySelector('.next_lesson_bg a, .next-btn, [class*="next"]');
    if (btn) {
      btn.click();
      log('点击下一节');
      setTimeout(doTask, 5000);
      return;
    }

    // 查找视频列表
    let items = document.querySelectorAll('.catalogue_ul1 li[id^="video-"], .file-item, [class*="chapter"], [class*="video-item"]');
    if (items.length === 0) {
      // 尝试其他选择器
      items = document.querySelectorAll('[id^="video-"], [data-id^="video-"]');
    }
    if (items.length === 0) { 
      log('未找到视频列表'); 
      return; 
    }

    // 找到当前播放的视频
    let currIdx = -1;
    let curr = document.querySelector('.catalogue_ul1 li.current, .active, [class*="current"], [class*="playing"]');
    items.forEach(function(itm, i) { 
      if (itm === curr || itm.classList.contains('current') || itm.classList.contains('active') || itm.classList.contains('playing')) {
        currIdx = i; 
      }
    });

    log('当前视频: ' + (currIdx + 1) + ', 总数: ' + items.length);

    // 查找下一个未完成的视频
    for (let i = currIdx + 1; i < items.length; i++) {
      let state = items[i].getAttribute('watchstate');
      let done = items[i].querySelector('.time_icofinish, .icofinish, [class*="finish"], [class*="done"]');
      // watchstate: 0=未学, 1=学习中, 2=已完成
      if (!done && (state === '0' || !state)) {
        items[i].click();
        log('播放第' + (i+1) + '个视频');
        setTimeout(doTask, 5000);
        return;
      }
    }
    log('全部完成');
    stop();
  }

  function handlePopup() {
    try {
      // 关闭按钮
      let closeBtn = document.querySelector('.popboxes_close, .pop-close, [class*="close"], .dialog-close');
      if (closeBtn) closeBtn.click();
      
      // iframe内的答题框
      let frame = document.querySelector('#tmDialog_iframe');
      if (frame) {
        try {
          let doc = frame.contentDocument || frame.contentWindow.document;
          let inputs = doc.querySelectorAll('input[type="radio"], input[type="checkbox"]');
          if (inputs.length > 0) {
            inputs[0].click();
            log('已选择答案');
          }
          let confirmBtn = doc.querySelector('.confirm-btn, .confirm, [class*="confirm"], .submit');
          if (confirmBtn) confirmBtn.click();
        } catch(e) {}
      }
    } catch(e) {}
  }

  // 监听消息
  chrome.runtime.onMessage.addListener(function(req, res) {
    log('收到: ' + req.action);
    if (req.action === 'ping') {
      res({status: 'pong'});
    } else if (req.action === 'start') { 
      start(); 
      res({status: 'ok'}); 
    } else if (req.action === 'stop') { 
      stop(); 
      res({status: 'ok'}); 
    }
    return true;
  });

  log('脚本就绪');
})();